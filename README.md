# End-of-Turn (EOT) Detection Assignment

## Overview

This project implements a lightweight End-of-Turn (EOT) detector using handcrafted prosodic features and a Logistic Regression classifier. The system predicts whether a pause corresponds to an End-of-Turn (EOT) or a Hold using only speech available before the pause.

---

## Repository Structure

```text
starter/
├── train.py
├── predict.py
├── features.py
├── RUNLOG.md
├── NOTES.md
├── SUMMARY.html
├── README.md
├── requirements.txt
└── predictions.csv
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

For English:

```bash
python train.py --data_dir ../eot_data/english --out predictions.csv
```

For Hindi:

```bash
python train.py --data_dir ../eot_data/hindi --out predictions.csv
```

This generates:

* `model.pkl`
* `scaler.pkl`
* `predictions.csv`

---

## Run Inference

```bash
python predict.py --data_dir <dataset_path> --model model.pkl --scaler scaler.pkl --out predictions.csv
```

Example (English):

```bash
python predict.py --data_dir ../eot_data/english --model model.pkl --scaler scaler.pkl --out predictions.csv
```

Example (Hindi):

```bash
python predict.py --data_dir ../eot_data/hindi --model model.pkl --scaler scaler.pkl --out predictions.csv
```

---

## Documentation

* **SUMMARY.html** – Project summary and methodology.
* **RUNLOG.md** – Experimental log and results.
* **NOTES.md** – Design decisions and implementation details.

---

## Author

Prashant Kumar
