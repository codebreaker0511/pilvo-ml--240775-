# RUNLOG

## Run 1 - English Baseline

**Commands**

```bash
python baseline.py --data_dir ../eot_data/english --out base.csv
python score.py --data_dir ../eot_data/english --pred base.csv
```

**Results**

* AUC: 0.514
* Mean response delay: 1600 ms
* Interrupted turns: 0.0%
* Operating threshold: 1.0

**Changes**

* Executed the provided silence-only baseline without any modifications.
* Verified the starter environment and scoring pipeline.

---

## Run 2 - Hindi Baseline

**Commands**

```bash
python baseline.py --data_dir ../eot_data/hindi --out base_hi.csv
python score.py --data_dir ../eot_data/hindi --pred base_hi.csv
```

**Results**

* AUC: 0.501
* Mean response delay: 850 ms
* Interrupted turns: 5.0%
* Operating threshold: 0.05

**Changes**

* Evaluated the provided baseline on the Hindi dataset.
* Confirmed the baseline and scorer work correctly on both languages.

---

## Run 3 - Starter Logistic Regression (English)

**Commands**

```bash
python train.py --data_dir ../eot_data/english --out mine.csv
python score.py --data_dir ../eot_data/english --pred mine.csv
```

**Results**

* Held-out turn accuracy: 0.600
* AUC: 0.599
* Mean response delay: 1190 ms
* Interrupted turns: 5.0%
* Operating threshold: 0.55

**Changes**

* Ran the provided Logistic Regression training pipeline using the starter prosodic features:

  * Mean energy near the pause
  * Final voiced pitch
  * Speech context duration
* Observed a reduction in response delay from 1600 ms to 1190 ms on the English development set.

---

## Run 4 - Starter Logistic Regression (Hindi)

**Commands**

```bash
python train.py --data_dir ../eot_data/hindi --out mine_hi.csv
python score.py --data_dir ../eot_data/hindi --pred mine_hi.csv
```

**Results**

* Held-out turn accuracy: 0.569
* AUC: 0.634
* Mean response delay: 850 ms
* Interrupted turns: 5.0%
* Operating threshold: 0.05

**Changes**

* Evaluated the same starter model on the Hindi dataset.
* Observed an improvement in AUC over the silence baseline while response delay remained unchanged.

## Run 5 – Experiment 1: Rich Prosodic Features (English)

### Objective

Improve the starter model by replacing the minimal 3-feature representation with a richer set of causal prosodic features while keeping the same Logistic Regression classifier.

### Changes Made

Modified `extract_features()` in `train.py`.

Replaced the original features:

* Mean energy of last few frames
* Mean final voiced pitch
* Speech duration

with the following 10-dimensional feature vector:

1. Mean frame energy
2. Energy standard deviation
3. Final frame energy
4. Energy slope (linear trend)
5. Mean pitch (voiced frames)
6. Pitch standard deviation
7. Final pitch
8. Pitch slope (linear trend)
9. Voiced-frame ratio
10. Speech duration

All features were extracted **strictly from audio before `pause_start`**, preserving causality.

### Motivation

The hypothesis was that end-of-turns exhibit richer prosodic patterns than captured by the starter features. In particular:

* decreasing energy
* falling pitch
* reduced voicing
* longer speech context

may better distinguish End-of-Turn (EOT) pauses from Hold pauses.

### Results

Training:

* Held-out turn accuracy: **0.631**

Evaluation (English):

* AUC: **0.604**
* Mean Response Delay: **1230 ms**
* Interrupted Turns: **5.0%**

### Comparison

| Model         |       AUC |       Delay |
| ------------- | --------: | ----------: |
| Starter Model |     0.599 | **1190 ms** |
| Experiment 1  | **0.604** | **1230 ms** |

### Observation

Although the richer feature set slightly improved AUC, it increased response delay by approximately **40 ms** compared to the starter model.

This suggests that the additional handcrafted features improved overall ranking performance but did not improve the operating point used under the assignment's 5% interruption constraint. Some of the added features may be redundant or introduce noise for a linear classifier.

### Conclusion

Experiment 1 **did not outperform** the starter model on the primary evaluation metric (response delay). The starter feature set remains the better-performing model so far.

## Run 6 – Experiment 2: Pause Position (pause_index)

### Motivation

The assignment hints mention:

> "speaking-rate context, position of the pause within the turn so far"

Our previous models relied entirely on acoustic features extracted from the speech preceding the pause. However, the metadata also contains `pause_index`, which indicates the number of pauses encountered so far within the current turn.

This feature is causal because, during real-time inference, the system always knows how many pauses have occurred before the current one.

Hypothesis:

* Later pauses in a turn may be more likely to correspond to the end of the speaker's turn (EOT).
* Logistic Regression may benefit from this additional contextual signal.

---

### Changes

Added `pause_index` as an additional feature.

Feature vector changed from:

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
duration
]

to

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
pause_index
]

---

### Results (English)

Held-out Accuracy: 0.585

AUC: 0.651

Best operating point (≤5% interrupted turns):

* Mean response delay: 1295 ms
* Interrupted turns: 5.0%

---

### Comparison

| Model                  |       AUC |       Delay |
| ---------------------- | --------: | ----------: |
| Starter                |     0.599 |     1190 ms |
| Rich Prosodic Features |     0.604 |     1230 ms |
| + StandardScaler       |     0.609 |     1272 ms |
| + pause_index          | **0.651** | **1295 ms** |

---

### Observation

Adding `pause_index` significantly improved AUC, indicating that the model ranked EOT pauses more accurately overall.

However, the assignment metric (mean response delay under a 5% interruption constraint) became worse. Although the classifier's ranking improved, it did not produce earlier confident EOT predictions at thresholds satisfying the interruption constraint.

This highlights an important distinction between optimizing AUC and optimizing the assignment's latency-based evaluation metric.

---

### Conclusion

`pause_index` improves overall discrimination but does not improve the primary evaluation metric.

The next experiments should focus on stronger acoustic cues immediately preceding the pause rather than additional global or metadata features.

## Run 7 – Experiment 4: Local Energy Features (Energy Decay into Pause)

### Motivation

The assignment explicitly suggests exploring:

> "energy decay rate into the pause"

Our previous features summarized energy statistics over the entire 1.5-second speech window preceding the pause. However, the final few hundred milliseconds before a pause are likely to contain stronger end-of-turn cues than earlier parts of the utterance.

Hypothesis:

* Energy immediately before an EOT pause should decay differently from energy before a HOLD pause.
* Local energy statistics should therefore provide stronger predictive power than global energy statistics.

---

### Changes

Retained all previous features and added four new energy features computed only from the final 300 ms of speech before the pause:

* local_energy_mean
* local_energy_std
* local_energy_last
* local_energy_slope

These were appended to the existing feature vector without removing any previous features.

---

### Results (English)

Held-out Accuracy: (record your printed value)

AUC: **0.702**

Best operating point (≤5% interrupted turns):

* Mean response delay: **1260 ms**
* Interrupted turns: **5.0%**

---

### Comparison

| Model                  |       AUC |       Delay |
| ---------------------- | --------: | ----------: |
| Starter                |     0.599 | **1190 ms** |
| Rich Prosodic Features |     0.604 |     1230 ms |
| + StandardScaler       |     0.609 |     1272 ms |
| + pause_index          |     0.651 |     1295 ms |
| + Local Energy         | **0.702** | **1260 ms** |

---

### Observation

Adding local energy features produced the largest AUC improvement observed so far, confirming that acoustic information immediately preceding the pause is substantially more informative than statistics computed across the entire speech segment.

Despite the improved discrimination, the latency metric did not surpass the starter baseline, indicating that stronger ranking alone is insufficient to produce earlier end-of-turn decisions under the 5% interruption constraint.

---

### Conclusion

This experiment validates the assignment's hypothesis that local acoustic behavior near the pause boundary is important.

Future experiments should extend this idea to pitch by analyzing the **last voiced region** rather than the entire utterance.

## Run 8 – Experiment 5: Local Prosodic Features (English + Hindi)

### Motivation

The assignment suggests that end-of-turn decisions depend primarily on prosodic cues immediately preceding the pause rather than statistics computed over the entire utterance.

Previous experiments showed that global energy and pitch statistics improved AUC only marginally. Therefore, this experiment focused on extracting **local acoustic features** near the pause boundary.

The following assignment hints were targeted:

* Energy decay into the pause
* F0 behavior over the last voiced region
* Position of the pause within the turn so far

---

### Changes

Starting from the previous feature set, the following features were added:

* Local energy mean (last 300 ms)

* Local energy standard deviation

* Local energy at the end of speech

* Local energy slope

* Local pitch mean (last voiced frames)

* Local pitch standard deviation

* Local pitch at the end of speech

* Local pitch slope

The existing contextual feature `pause_index` was retained.

No changes were made to the classifier or evaluation pipeline.

---

## English Results

Held-out Accuracy: (record printed value)

AUC: **0.717**

Best operating point (≤5% interrupted turns):

* Mean response delay: **1250 ms**
* Interrupted turns: **5.0%**

---

## Hindi Results

Held-out Accuracy: **0.569**

AUC: **0.769**

Best operating point (≤5% interrupted turns):

* Mean response delay: **850 ms**
* Interrupted turns: **5.0%**

---

### Observations

The addition of local prosodic features consistently improved ranking performance compared to earlier experiments.

Most notably, the Hindi dataset achieved a substantially higher AUC (0.769) together with a much lower response delay (850 ms), indicating that local energy and pitch cues are highly informative for Hindi turn-taking.

The English dataset also showed continuous AUC improvement (0.599 → 0.717), although this improvement did not translate into a lower latency than the starter baseline.

---

### Conclusion

This experiment confirms that acoustic cues immediately preceding a pause are significantly more informative than global utterance-level statistics.

The strong performance on Hindi suggests that local prosodic behavior is an effective feature representation for end-of-turn prediction.

Future work should investigate the remaining assignment hint:

* Final-syllable lengthening (duration of the final voiced stretch relative to the speaker's earlier speech)

as a complementary temporal feature.

# Run #9 – Refining Local Pitch & Final-Syllable Lengthening

## Objective

Improve end-of-turn detection by implementing the remaining prosodic cues suggested in the assignment handout:

1. Better representation of the speaker's final voiced region.
2. Final-syllable lengthening using voiced run statistics.

---

## Experiment 6 – Actual Last Voiced Region

### Motivation

Previously, local pitch statistics were computed from the last few voiced frames, regardless of whether they belonged to the same continuous voiced segment.

To better match the assignment hint ("last voiced region"), the feature extraction was modified to locate the final continuous voiced region in the F0 contour and compute local pitch statistics only from that segment.

### Changes

* Scanned the F0 contour backwards.
* Ignored trailing unvoiced frames.
* Extracted the final continuous voiced segment.
* Computed:

  * local_pitch_mean
  * local_pitch_std
  * local_pitch_last
  * local_pitch_slope

### English Results

AUC: **0.707**

Mean response delay: **1265 ms**

Interrupted turns: **4.0%**

### Observation

Although this implementation more closely followed the assignment description, it performed worse than using the previous approximation (last several voiced frames). Therefore, the change was not retained as an improvement by itself.

---

## Experiment 7 – Final-Syllable Lengthening

### Motivation

The assignment notes that speakers often lengthen the final syllable before completing a turn.

Instead of relying only on pitch and energy, timing-based features were added to explicitly model this phenomenon.

### New Features

Computed voiced run statistics from the F0 contour:

* last_run_length
* average_previous_run_length
* lengthening_ratio
* last_run_fraction

These features estimate whether the final voiced segment is unusually long compared to the speaker's earlier speech.

### English Results

AUC: **0.722**

Mean response delay: **1130 ms**

Interrupted turns: **5.0%**

Operating threshold: **0.6**

### Hindi Results

AUC: **0.774**

Mean response delay: **830 ms**

Interrupted turns: **5.0%**

Operating threshold: **0.5**

---

## Conclusion

Adding explicit timing-based prosodic features successfully improved performance on both evaluation datasets.

Compared to the previous best system:

### English

* AUC improved from **0.717 → 0.722**
* Mean response delay reduced from **1250 ms → 1130 ms**

### Hindi

* AUC improved from **0.769 → 0.774**
* Mean response delay reduced from **850 ms → 830 ms**

The consistent improvement across both languages indicates that modeling final-syllable lengthening provides complementary information beyond energy and pitch statistics alone.
