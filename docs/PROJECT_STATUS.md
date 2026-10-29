# PROJECT STATUS
## Real-Time Sign Language to Text Conversion System
**Institution:** SRM Institute of Science and Technology  
**Last updated:** 2026-09-23

---

## Current Phase: DATA COLLECTION (Phase 4 infrastructure ✅ — awaiting your collection)

| Phase | Name | Status | Tests |
|---|---|---|---|
| Phase 1 | Environment, Config, Core Types | ✅ Complete | 6/6 |
| Phase 2 | Camera + Hand Detection | ✅ Complete | 9/9 |
| Phase 3 | Feature Engineering & Normalization | ✅ Complete | 18/18 |
| Phase 4 | Data Collection Infrastructure | ✅ Complete (scripts ready) | 18/18 |
| **Phase 4** | **Dataset Collection (YOU)** | **⏳ IN PROGRESS** | — |
| Phase 5 | Dataset Splitting | ⬜ Waiting for data | — |
| Phase 6 | Model Training (6 candidates) | ⬜ Waiting | — |
| Phase 7 | Model Evaluation | ⬜ Waiting | — |
| Phase 8 | Hyperparameter Tuning | ⬜ Waiting | — |
| Phase 9 | Real-Time Inference Pipeline | ⬜ Waiting | — |
| Phase 10 | Prediction Stabilization | ⬜ Waiting | — |
| Phase 11 | Text Composition | ⬜ Waiting | — |
| Phase 12 | Final Integration | ⬜ Waiting | — |
| Phase 13 | Benchmarking | ⬜ Waiting | — |

**Total automated tests:** 51/51 passing  
**Feature vector:** (63,) float32, version `v1-base-63`  
**Vocabulary size:** 13 classes (12 signs + _NEUTRAL)

---

## Dataset Collection Progress

| Class | Samples | Sessions | Status |
|---|---|---|---|
| HELLO | 0 | 0 | ⬜ Not started |
| THANK_YOU | 0 | 0 | ⬜ Not started |
| PLEASE | 0 | 0 | ⬜ Not started |
| YES | 0 | 0 | ⬜ Not started |
| NO | 0 | 0 | ⬜ Not started |
| SORRY | 0 | 0 | ⬜ Not started |
| HELP | 0 | 0 | ⬜ Not started |
| NAME | 0 | 0 | ⬜ Not started |
| WATER | 0 | 0 | ⬜ Not started |
| EAT | 0 | 0 | ⬜ Not started |
| I_LOVE_YOU | 0 | 0 | ⬜ Not started |
| YOU | 0 | 0 | ⬜ Not started |
| _NEUTRAL | 0 | 0 | ⬜ Not started |
| **TOTAL** | **0** | **0** | — |

> **Update this table** after each collection session using `--status` output.  
> Minimum to proceed to Phase 5: **200 samples × 4 sessions per class**

---

## Key Documents

| Document | Purpose |
|---|---|
| [SIGN_VOCABULARY.md](SIGN_VOCABULARY.md) | Sign specifications, collection instructions, confusion risks |
| [DATA_COLLECTION_SOP.md](DATA_COLLECTION_SOP.md) | Step-by-step collection procedure |
| [TRAINING_SOP.md](TRAINING_SOP.md) | Full training pipeline procedure |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Record of all ML experiments |
| [DATASET_LOG.md](DATASET_LOG.md) | Dataset collection log |
| [VIVA_NOTES.md](VIVA_NOTES.md) | Viva preparation questions and answers |
| [PROGRESS_REPORT.md](PROGRESS_REPORT.md) | Technical implementation progress |
| [TEST_RESULTS.md](TEST_RESULTS.md) | Automated test history |
