import argparse
import csv
import os

import joblib
import numpy as np

from features import (
    load_wav,
    speech_before,
    frame_energy_db,
    f0_contour,
)


def extract_features(x, sr, pause_start, pause_index):
    """Extract the exact same 23 features used during training."""

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

    # ---------- Local Energy ----------

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
            1,
        )[0]
    else:
        local_energy_slope = 0.0

    # ---------- Pitch ----------

    if len(voiced) > 0:

        pitch_mean = np.mean(voiced)
        pitch_std = np.std(voiced)
        pitch_last = voiced[-1]

        if len(voiced) > 1:
            pitch_slope = np.polyfit(
                np.arange(len(voiced)),
                voiced,
                1,
            )[0]
        else:
            pitch_slope = 0.0

    else:

        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_last = 0.0
        pitch_slope = 0.0

    # ---------- Last continuous voiced region ----------

    i = len(f0) - 1

    while i >= 0 and f0[i] == 0:
        i -= 1

    if i >= 0:

        end = i

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
                1,
            )[0]
        else:
            local_pitch_slope = 0.0

    else:

        local_pitch_mean = 0.0
        local_pitch_std = 0.0
        local_pitch_last = 0.0
        local_pitch_slope = 0.0

    voiced_ratio = len(voiced) / len(f0) if len(f0) else 0.0

    return np.array(
        [
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
            last_run_fraction,
        ],
        dtype=np.float32,
    )

def predict_folder(data_dir, model_path, scaler_path, out_path):
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    labels_path = os.path.join(data_dir, "labels.csv")

    rows = list(csv.DictReader(open(labels_path)))

    cache = {}

    X = []
    keys = []

    for r in rows:

        wav_path = os.path.join(data_dir, r["audio_file"])

        if wav_path not in cache:
            cache[wav_path] = load_wav(wav_path)

        x, sr = cache[wav_path]

        feat = extract_features(
            x,
            sr,
            float(r["pause_start"]),
            int(r["pause_index"]),
        )

        X.append(feat)
        keys.append((r["turn_id"], r["pause_index"]))

    X = np.asarray(X, dtype=np.float32)

    X = scaler.transform(X)

    probs = clf.predict_proba(X)[:, 1]

    with open(out_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "turn_id",
                "pause_index",
                "p_eot",
            ]
        )

        for (turn_id, pause_index), p in zip(keys, probs):
            writer.writerow(
                [
                    turn_id,
                    pause_index,
                    f"{p:.4f}",
                ]
            )

    print(f"Wrote {len(probs)} predictions -> {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Folder containing labels.csv and audio files",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (.pkl)",
    )

    parser.add_argument(
        "--scaler",
        required=True,
        help="Path to fitted scaler (.pkl)",
    )

    parser.add_argument(
        "--out",
        default="predictions.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    predict_folder(
        data_dir=args.data_dir,
        model_path=args.model,
        scaler_path=args.scaler,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()