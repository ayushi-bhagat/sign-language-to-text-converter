# MODEL_RESULTS.md — Model Selection & Evaluation Metrics

**Project:** Real-Time Sign Language to Text Conversion System  
**Status:** No model trained yet (Training scheduled for Phase 6, Evaluation for Phase 7)  

---

## 1. Candidate Model Selection (Phase 6)

* **Cross-Validation Scheme:** 5-Fold `StratifiedGroupKFold` grouped by `session_id`.
* **Primary Metric:** Macro-F1.

| Candidate Algorithm | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean Macro-F1 | Std Dev | Winner |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |
| K-Nearest Neighbors | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |
| Support Vector Machine (RBF) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |
| Random Forest | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |
| Extra Trees | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |
| Multi-Layer Perceptron (MLP) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | ⬜ |

---

## 2. Final Winning Model Test Evaluation (Phase 7)

* **Selected Model:** `TBD — Phase 6`
* **Evaluation Data:** Held-out test sessions (never seen during CV or hyperparameter search).
* **Test Performance Summary:**
  * **Macro F1:** `TBD — Phase 7`
  * **Weighted F1:** `TBD — Phase 7`
  * **Overall Accuracy:** `TBD — Phase 7`
  * **Balanced Accuracy:** `TBD — Phase 7`
  * **Macro Precision:** `TBD — Phase 7`
  * **Macro Recall:** `TBD — Phase 7`

---

## 3. Per-Class Performance Table (Phase 7)

| Class | Support (Samples) | Precision | Recall | F1-Score | Most Common Confusion |
|---|---|---|---|---|---|
| `HELLO` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `THANK_YOU` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `PLEASE` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `YES` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `NO` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `SORRY` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `HELP` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `NAME` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `WATER` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `EAT` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `I_LOVE_YOU` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `YOU` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| `_NEUTRAL` | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

---

## 4. Model Artifact Provenance

* **Classifier Artifact:** `models/v1/sign_classifier.joblib` (`TBD`)
* **Label Encoder Artifact:** `models/v1/label_encoder.joblib` (`TBD`)
* **Metadata Manifest:** `models/v1/model_metadata.json` (`TBD`)
