# NOTES.md

## Overview

This submission implements a lightweight, fully causal End-of-Turn (EOT) detector using handcrafted prosodic features and a Logistic Regression classifier.

The implementation strictly follows the assignment constraint of using only the speech available before `pause_start`. No future information (such as pause duration or speech after the pause) is used during feature extraction.

---

## Model

- StandardScaler
- Logistic Regression
- `class_weight="balanced"`
- `max_iter=1000`

The model was chosen because it is:
- Simple and interpretable
- Fast to train
- Suitable for small datasets
- Provides calibrated probabilities using `predict_proba()`

---

## Final Feature Set (23 Features)

### Global Energy
- Mean energy
- Standard deviation
- Last frame energy
- Energy slope

### Global Pitch
- Mean F0
- Standard deviation
- Last voiced pitch
- Pitch slope

### Voice Activity
- Voiced ratio
- Speech duration
- Pause index

### Local Energy (Last 300 ms)
- Mean energy
- Standard deviation
- Last frame energy
- Energy slope

### Local Pitch
Instead of using the last fixed number of voiced frames, the implementation detects the actual final continuous voiced region before the pause and computes:
- Mean pitch
- Standard deviation
- Last pitch
- Pitch slope

### Final-Syllable Lengthening
To capture natural conversational turn endings, four additional duration-based features were added:
- Last voiced run length
- Average previous voiced run length
- Lengthening ratio
- Fraction of utterance occupied by the last voiced run

---

## Experiments Performed

### Baseline
Used the starter implementation with basic energy and pitch statistics.

### Energy Slope
Added energy decay information approaching the pause.

### Pause Index
Included pause position within the utterance as contextual information.

### Local Energy Features
Computed energy statistics over only the final 300 ms before the pause.

### Last Continuous Voiced Region
Instead of computing local pitch from the last few voiced frames, pitch statistics were extracted from the actual final voiced segment.

### Final-Syllable Lengthening
Added four duration-based features motivated by phonetic observations that speakers often lengthen the final syllable before yielding a turn.

These features produced the final submitted model.

---

## Design Decisions

The implementation intentionally avoids:

- Pause duration
- Audio after the pause
- Future transcript information
- Lexical information
- Language-specific rules

This keeps the detector completely causal and suitable for real-time deployment.

---

## Limitations

Some errors remain due to:

- Noisy pitch estimation
- Breath noises near pauses
- Speaker variability
- Emotional speech
- Overlap between hold pauses and true turn endings

---

## Possible Future Improvements

Given more time, the following extensions would likely improve performance:

- Gradient Boosted Trees (LightGBM/XGBoost)
- Better pitch extraction (CREPE or Praat)
- Speaking-rate estimation
- Spectral tilt features
- Temporal sequence models (LSTM/Transformer)
- Larger multilingual training corpus

---

## Files

- `train.py` — training pipeline
- `predict.py` — inference on unseen data
- `features.py` — audio utilities and feature extraction
- `RUNLOG.md` — experiment history
- `SUMMARY.html` — project summary