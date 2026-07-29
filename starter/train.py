"""Skeleton: prosodic features + classifier. Runs as-is, scores poorly ON
PURPOSE. Your hour goes into extract_features() and what you learn from
your errors.

    python train.py --data_dir eot_data/english --out predictions.csv

Ideas worth testing (this is the assignment, not a checklist):
  - F0 slope over the last voiced region (statements fall, continuations
    often stay level or rise)
  - final-syllable lengthening: last voiced stretch duration vs the
    speaker's average
  - energy decay rate into the pause
  - speaking-rate context, position of the pause within the turn so far
  - anything you discover by LISTENING to your misclassified pauses
"""
import argparse
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from features import load_wav, speech_before, frame_energy_db, f0_contour


def extract_features(x, sr, pause_start, pause_index):
    """Features from audio STRICTLY BEFORE pause_start."""

    seg = speech_before(x, sr, pause_start, window_s=1.5)

    if len(seg) < sr // 10:
        return np.zeros(23, dtype=np.float32)
    
    duration = len(seg) / sr
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)

    voiced = f0[f0 > 0]

    # ---------- Final-syllable lengthening ----------

    run_lengths = []
    cur = 0

    for v in f0:
        if v > 0:
            cur += 1
        else:
            if cur > 0:
                run_lengths.append(cur)
            cur = 0

    if cur > 0:
        run_lengths.append(cur)

    if len(run_lengths) > 0:

        last_run_length = float(run_lengths[-1])

        if len(run_lengths) > 1:
            avg_previous_run = float(np.mean(run_lengths[:-1]))
        else:
            avg_previous_run = last_run_length

        lengthening_ratio = last_run_length / (avg_previous_run + 1e-6)

        last_run_fraction = last_run_length / max(len(f0), 1)

    else:

        last_run_length = 0.0
        avg_previous_run = 0.0
        lengthening_ratio = 0.0
        last_run_fraction = 0.0



    # ---------- Energy ----------
    energy_mean = np.mean(e)
    energy_std = np.std(e)
    energy_last = e[-1]

    if len(e) > 1:
        energy_slope = np.polyfit(np.arange(len(e)), e, 1)[0]
    else:
        energy_slope = 0.0

    # ---------- Local energy (last 300 ms) ----------

    last_ms = 0.3
    n_frames = max(2, int(len(e) * last_ms / duration))

    e_local = e[-n_frames:]

    local_energy_mean = np.mean(e_local)
    local_energy_std = np.std(e_local)
    local_energy_last = e_local[-1]

    if len(e_local) > 1:
        local_energy_slope = np.polyfit(
            np.arange(len(e_local)),
            e_local,
            1
        )[0]
    else:
        local_energy_slope = 0.0

    # ---------- Pitch ----------
    if len(voiced) > 0:
        pitch_mean = np.mean(voiced)
        pitch_std = np.std(voiced)
        pitch_last = voiced[-1]

        if len(voiced) > 1:
            pitch_slope = np.polyfit(np.arange(len(voiced)), voiced, 1)[0]
        else:
            pitch_slope = 0.0
    else:
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_last = 0.0
        pitch_slope = 0.0

    # ---------- Local Pitch (actual last voiced region) ----------

    # Find the last continuous voiced region in the F0 contour.

    i = len(f0) - 1

    # Skip trailing unvoiced frames.
    while i >= 0 and f0[i] == 0:
        i -= 1

    if i >= 0:

        end = i

        # Walk backwards until the voiced region begins.
        while i >= 0 and f0[i] > 0:
            i -= 1

        start = i + 1

        last_voiced = f0[start:end + 1]

        local_pitch_mean = np.mean(last_voiced)
        local_pitch_std = np.std(last_voiced)
        local_pitch_last = last_voiced[-1]

        if len(last_voiced) > 1:
            local_pitch_slope = np.polyfit(
                np.arange(len(last_voiced)),
                last_voiced,
                1
            )[0]
        else:
            local_pitch_slope = 0.0

    else:
        local_pitch_mean = 0.0
        local_pitch_std = 0.0
        local_pitch_last = 0.0
        local_pitch_slope = 0.0

    # ---------- Voice activity ----------
    voiced_ratio = len(voiced) / len(f0) if len(f0) else 0.0

    return np.array([
        energy_mean,
        energy_std,
        energy_last,
        energy_slope,

        pitch_mean,
        pitch_std,
        pitch_last,
        pitch_slope,

        voiced_ratio,
        duration,
        float(pause_index),

        local_energy_mean,
        local_energy_std,
        local_energy_last,
        local_energy_slope,

        local_pitch_mean,
        local_pitch_std,
        local_pitch_last,
        local_pitch_slope,

        last_run_length,
        avg_previous_run,
        lengthening_ratio,
        last_run_fraction
    ], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))

    cache = {}
    X, y, groups, keys = [], [], [], []

    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])

        if path not in cache:
            cache[path] = load_wav(path)

        x, sr = cache[path]

        X.append(
            extract_features(
                x,
                sr,
                float(r["pause_start"]),
                int(r["pause_index"])
            )
        )
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))

    X = np.array(X)
    y = np.array(y)

    tr, te = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=0
        ).split(X, y, groups)
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X[tr])
    X_test = scaler.transform(X[te])

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    clf.fit(X_train, y[tr])

    print(
        f"held-out turn accuracy: {clf.score(X_test, y[te]):.3f} "
        f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})"
    )

    # Retrain on all data
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X)

    clf.fit(X_all, y)

    p = clf.predict_proba(X_all)[:, 1]

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])

        for (tid, pi), prob in zip(keys, p):
            w.writerow([tid, pi, f"{prob:.4f}"])

    print(f"wrote {len(keys)} predictions -> {args.out}")
    print(
        "NOTE for your final predict.py: it must load a SAVED model "
        "and predict on unseen data, not refit like this sanity script."
    )


if __name__ == "__main__":
    main()