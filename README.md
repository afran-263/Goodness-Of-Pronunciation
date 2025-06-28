Dysarthric Speech Goodness of Pronunciation (GoP) Evaluation
This project evaluates the Goodness of Pronunciation (GoP) for dysarthric speech using a phoneme classifier trained on healthy speech. It leverages Wav2Vec2 embeddings, Montreal Forced Aligner (MFA) for alignment, and a neural network for phoneme classification and scoring.

Project Structure
gop.py — Main script for training the phoneme recognizer, aligning dysarthric audio, and computing GoP scores.
phone_classifier.pt — Trained PyTorch model for phoneme classification.
id2phone.json — Mapping from numeric phoneme IDs to phoneme labels.
phone2id.json — Reverse mapping from phoneme labels to numeric IDs.

Features
Phoneme Classifier using Wav2Vec2-based features
Automatic forced alignment using Montreal Forced Aligner (MFA)
Goodness of Pronunciation (GoP) computation
Additional metrics: Entropy, Logit Margin, Max Logit, and Confidence Margin
Handles real-time input of dysarthric .wav files

Dataset
Healthy speech should follow the structure:
dataset/
└── TNI/
    ├── wav/
    │   └── file1.wav
    └── textgrid/
        └── file1.TextGrid
