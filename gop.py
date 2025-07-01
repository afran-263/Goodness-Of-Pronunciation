import os
import re
import glob
import json
import torch
import shutil
import textgrid
import subprocess
import torchaudio
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000
DATASET_DIR = "dataset"
MODEL_PATH = "phone_classifier.pt"
PH2ID_JSON = "phone2id.json"
ID2PH_JSON = "id2phone.json"

# === Step 1: Train phoneme recognizer using healthy speech ===
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
feature_extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
feature_extractor.eval()

phone2id = {}
id2phone = {}

def update_phone_dict(phone):
    if phone not in phone2id:
        idx = len(phone2id)
        phone2id[phone] = idx
        id2phone[idx] = phone

class PhoneDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

if os.path.exists(MODEL_PATH):
    print("model exists")
else:
    print("model not found, training...")
    def extract_phoneme_data(wav_path, tg_path):
        print(f"Processing {wav_path} and {tg_path}")
        waveform, sr = torchaudio.load(wav_path)
        waveform = torchaudio.transforms.Vol(1.1)(waveform)
        if sr != SAMPLING_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

        inputs = processor(waveform.squeeze(), sampling_rate=SAMPLING_RATE, return_tensors="pt").input_values.to(DEVICE)
        with torch.no_grad():
            features = feature_extractor(inputs).last_hidden_state.squeeze(0).cpu().numpy()
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6) # Average over time
        
        grid = textgrid.TextGrid.fromFile(tg_path)
        phone_tier = next((t for t in grid.tiers if t.name.lower() in ["phones", "phoneme"]), grid.tiers[0])

        duration = waveform.shape[1] / SAMPLING_RATE
        phoneme_data = []

        for interval in phone_tier.intervals:
            label = interval.mark.strip()
            if label in ["", "sp", "sil"]: continue
            update_phone_dict(label)
            start_idx = int((interval.minTime / duration) * features.shape[0])
            end_idx = int((interval.maxTime / duration) * features.shape[0])
            if end_idx <= start_idx or end_idx > features.shape[0]:
                print(f"Skipping {label} with invalid indices.")
                continue
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
            avg_feat = features[start_idx:end_idx].mean(axis=0)
            phoneme_data.append((avg_feat, phone2id[label]))

        return phoneme_data

    all_data = []
    for spk in ["TNI", "RRBI", "SVBI"]:
        wav_dir = os.path.join(DATASET_DIR, spk, "wav")
        tg_dir = os.path.join(DATASET_DIR, spk, "textgrid")
        for wav_file in glob.glob(os.path.join(wav_dir, "*.wav")):
            file_id = os.path.splitext(os.path.basename(wav_file))[0]
            tg_file = os.path.join(tg_dir, file_id + ".TextGrid")
            if os.path.exists(tg_file):
                all_data.extend(extract_phoneme_data(wav_file, tg_file))

    print("Phoneme frequency:", Counter([y for _, y in all_data]))
    with open(PH2ID_JSON, "w") as f: json.dump(phone2id, f)
    with open(ID2PH_JSON, "w") as f: json.dump(id2phone, f)

    dataset = PhoneDataset(all_data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    class PhoneClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, len(phone2id))
            )

        def forward(self, x): return self.fc(x)

    model = PhoneClassifier().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)

# === Upload dysarthric speech ===
dysarthric_path = input("Enter path to dysarthric audio (WAV format): ").strip()
if not os.path.exists(dysarthric_path):
    raise FileNotFoundError(f"File not found: {dysarthric_path}")
if not dysarthric_path.lower().endswith(".wav"):
    raise ValueError("Please provide a valid .wav audio file.")

shutil.copy(dysarthric_path, "dysarthric.wav")  # standardize filename for later steps

transcript = input("Enter transcript (exact text spoken in the uploaded audio): ").strip()
with open("dysarthric.txt", "w", encoding="utf-8") as f:
    f.write(transcript)


def normalize_transcript(text):
    return re.sub(r"[^\w\s']", "", text.lower()).strip()

def run_mfa():
    tmp = "temp_align"
    os.makedirs(tmp, exist_ok=True)
    aligned = os.path.join(tmp, "aligned")
    os.makedirs(aligned, exist_ok=True)
    shutil.copy("dysarthric.wav", os.path.join(tmp, "realtime.wav"))
    with open(os.path.join(tmp, "realtime.txt"), "w", encoding="utf-8") as f:
        f.write(normalize_transcript(transcript))

    cmd = [
        r"C:\MFA\montreal-forced-aligner\bin\mfa_align.exe",
        tmp,
        r"C:\MFA\model\english.dict.txt",
        r"C:\MFA\model\english.zip",
        aligned,
        "--clean", "--debug"
    ]
    subprocess.run(cmd)
    return os.path.join("temp_align", "aligned", "temp_align", "realtime.TextGrid")

textgrid_path = run_mfa()

# === GoP Scoring ===
class PhoneClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(phone2id))
        )

    def forward(self, x): return self.fc(x)

with open(ID2PH_JSON) as f: id2phone = json.load(f)
phone2id = {v: int(k) for k, v in id2phone.items()}
classifier = PhoneClassifier().to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()

waveform, sr = torchaudio.load("dysarthric.wav")
if sr != SAMPLING_RATE:
    waveform = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(waveform)

if waveform.shape[0] > 1:  # convert stereo to mono
    waveform = waveform.mean(dim=0, keepdim=True)

inputs = processor(
    waveform.squeeze(0),  # shape: [time]
    sampling_rate=SAMPLING_RATE,
    return_tensors="pt"
).input_values.to(DEVICE)

with torch.no_grad():
    features = feature_extractor(inputs).last_hidden_state.squeeze(0).cpu().numpy()
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
duration = waveform.shape[1] / SAMPLING_RATE

tg = textgrid.TextGrid.fromFile(textgrid_path)
tier = next((t for t in tg.tiers if t.name.lower() in ["phones", "phoneme"]), tg.tiers[0])
scores = []

for intv in tier.intervals:
    label = intv.mark.strip()
    if label.lower() in ["", "sp", "sil"] or label not in phone2id:
        continue
    s_idx = int((intv.minTime / duration) * features.shape[0])
    e_idx = int((intv.maxTime / duration) * features.shape[0])
    if e_idx <= s_idx: continue

    feat = torch.tensor(features[s_idx:e_idx].mean(axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(feat)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
    correct = phone2id[label]
    gop = probs[correct] - np.max(np.delete(probs, correct))
    predicted = id2phone[str(np.argmax(probs))]
    scores.append((label, predicted, round(gop, 3), (intv.minTime, intv.maxTime)))
    

print("\nPhoneme-wise Scores and Predictions:")
print(f"{'Actual':<6} {'Pred':<6} {'GoP':>6} {'Entropy':>8} {'Margin':>8} {'MaxLogit':>9}    {'LogitMargin':>13}")
print("-" * 60)

for intv in tier.intervals:
    label = intv.mark.strip()
    if label.lower() in ["", "sp", "sil"] or label not in phone2id:
        continue

    s_idx = int((intv.minTime / duration) * features.shape[0])
    e_idx = int((intv.maxTime / duration) * features.shape[0])
    if e_idx <= s_idx: continue

    feat = torch.tensor(features[s_idx:e_idx].mean(axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier(feat).squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        logits = logits.cpu().numpy()

    correct_id = phone2id[label]
    predicted_id = int(np.argmax(probs))
    predicted = id2phone[str(predicted_id)]

    gop = probs[correct_id] - np.max(np.delete(probs, correct_id))

    entropy = -np.sum(probs * np.log(probs + 1e-9))
    top2 = np.partition(probs, -2)[-2:]
    margin = top2[-1] - top2[-2]
    max_logit = logits[correct_id]
    logit_others = np.delete(logits, correct_id)
    logit_margin = max_logit - np.max(logit_others)


    scores.append((label, predicted, round(gop, 3), (intv.minTime, intv.maxTime)))

    print(f"{label:<6} {predicted:<6} {gop:6.3f} {entropy:8.3f} {margin:8.3f} {max_logit:9.3f} {logit_margin:13.3f}")

print(f"\nOverall Mean GoP Score: {np.mean([s for _, _, s, _ in scores]):.3f}")
