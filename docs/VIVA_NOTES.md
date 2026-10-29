# VIVA NOTES
## Real-Time Sign Language to Text Conversion System

> ⚠️ **Important:** The answers below are FRAMEWORK answers. After you run experiments, update each answer with your ACTUAL numbers and observations. A viva answer citing real experimental results is far stronger than a memorized theoretical answer.

---

## Category 1 — System Architecture

### Q1. Why did you choose MediaPipe for hand detection?

**Core answer:**  
MediaPipe Hands is a production-grade open-source library by Google that detects 21 anatomical hand landmarks in 3D from a monocular webcam at 25-30fps on CPU. It eliminates the need to train a hand detector from scratch — which would require a large separate annotated dataset. It handles different skin tones, partial occlusion, and multiple hand sizes reliably.

**Supporting detail:**  
The alternative would be training a custom CNN hand detector, which requires a large bounding-box annotated dataset and weeks of training. MediaPipe is academically legitimate because we are building on top of an established detection baseline to focus on the *classification* problem.

---

### Q2. Why exactly 21 landmarks?

**Core answer:**  
The human hand has 21 major anatomical joint positions: the wrist (1) plus 4 joints per finger (5 fingers × 4 = 20) = 21 total. These 21 points form the minimal complete skeletal representation of all 5 fingers. Fewer points would lose information about which finger joints are flexed vs extended.

**Supporting detail:**  
The 21 landmarks follow the MediaPipe Hand Topology. Landmark 0 is the wrist. Landmarks 1-4 are the thumb (CMC, MCP, IP, TIP). Landmarks 5-8 are the index finger, and so on.

---

### Q3. Why 63 features? Why not more or fewer?

**Core answer:**  
21 landmarks × 3 coordinates each (x, y, z) = 63. This is the complete description of the 3D hand shape. After wrist normalization, all 63 values describe the relative position of each joint with respect to the wrist. Fewer features would lose hand shape information; more features would be redundant (there are only 21 anatomical joints to describe).

---

### Q4. Why normalize around the wrist?

**Core answer:**  
The wrist is the natural anatomical anchor of the hand. It is always detected as landmark 0 by MediaPipe. Subtracting the wrist position from all other landmarks removes the effect of where in the frame the hand appears — whether I hold my hand on the left or right side of the frame, after wrist subtraction, the sign HELLO produces the same feature values. This is called **translation invariance**.

**Intuition:** Imagine drawing the same hand pose on different positions of a whiteboard. After subtracting the starting point, all drawings overlap exactly.

---

### Q5. Why scale normalization?

**Core answer:**  
After wrist subtraction, the magnitude of the landmark coordinates still depends on how close the hand is to the camera — a nearby hand produces larger coordinate values. Dividing by the maximum landmark-to-wrist distance normalizes all hands to the same size, regardless of camera distance. This is **scale invariance**. After normalization, the furthest landmark from the wrist is always at distance 1.0.

---

### Q6. Why NOT normalize rotation?

**Core answer:**  
Two signs in our vocabulary — PLEASE (palm facing upward) and THANK_YOU (palm facing the camera) — differ ONLY in palm orientation. Rotation normalization would align both palms to a canonical orientation, making PLEASE and THANK_YOU produce identical feature vectors. The model could not learn to distinguish them. This decision is documented as ADR-004 in the project architecture and enforced by a config validation check.

---

### Q7. Why use classical ML instead of a CNN end-to-end?

**Core answer:**  
Four reasons:  
1. **Input structure:** Our input is already a 63-D normalized feature vector, not a raw image. CNNs process spatial pixel data — using a CNN on a 63-D vector is inappropriate.  
2. **Dataset size:** We have approximately [TBD] samples — too small for a deep CNN to generalize reliably without extensive transfer learning.  
3. **Interpretability:** Classical ML models (especially Random Forest) produce feature importances and clear confusion matrices. We can understand why errors occur.  
4. **Inference speed:** sklearn classifiers predict in under 1ms. A CNN would require GPU acceleration for real-time use.

---

## Category 2 — Machine Learning Choices

### Q8. Why compare six models?

**Core answer:**  
Different model families make different assumptions about the data structure. We don't know a priori whether our class boundaries are linear (Logistic Regression) or non-linear (SVM-RBF, MLP), or whether decision-tree ensemble methods would capture the pattern best. Empirical comparison is more rigorous than guessing.

---

### Q9. Why SVM with RBF kernel?

**Core answer:**  
SVM with RBF kernel finds the maximum-margin decision boundary in a high-dimensional feature space derived by the kernel function. It is historically one of the strongest classical models for medium-dimensional normalized feature spaces. The RBF kernel can model non-linear boundaries without specifying the explicit transformation.

**Your result:** [TBD — fill after EXP-01]

---

### Q10. Why Random Forest?

**Core answer:**  
Random Forest is an ensemble of 100-500 decision trees. Each tree is trained on a random bootstrap sample of data and considers a random subset of features at each split. The ensemble majority vote reduces overfitting compared to a single tree. Random Forest also produces **feature importances** — we can identify which of the 63 landmark coordinates are most discriminative.

**Your feature importance result:** [TBD — fill after EXP-01 with the top 5 most important features]

---

### Q11. Why grouped cross-validation?

**Core answer:**  
Standard cross-validation randomly assigns individual frames to folds. Frames from the same session would appear in both training and validation folds — causing data leakage (the leakage problem described in Q12). Grouped cross-validation ensures all frames from one session_id stay in the same fold, producing an honest estimate of generalization.

---

## Category 3 — Data Issues

### Q12. What is data leakage? Give a specific example.

**Core answer:**  
Data leakage occurs when information from the test set influences the model during training, producing an optimistically biased performance estimate.

**Specific example in this project:**  
If we randomly split 540 frames of HELLO across train/test, frame 47 (train) and frame 48 (test) are 1/8 second apart — nearly identical hand poses. The model trained on frame 47 has essentially already "seen" frame 48. Test accuracy reports 97%.

**Our demonstration (EXP-02):**  
Random split accuracy: [TBD]  
Session-grouped accuracy: [TBD]  
The difference is the magnitude of the leakage.

---

### Q13. Why can random frame splitting be misleading?

**Core answer:**  
At 8 fps, adjacent frames in the same session are 125ms apart. The hand pose changes negligibly in 125ms. This means most test frames have a nearly-identical twin in the training set. The model doesn't need to generalize — it just needs to recognize a pattern it has already memorized from an adjacent frame.

---

### Q14. Why is _NEUTRAL necessary?

**Core answer:**  
Without _NEUTRAL, the model is a **closed-world classifier** — it MUST assign every detected hand to one of the 12 sign classes. If you wave your hand, rest it, or transition between signs, the model would continuously produce false predictions. _NEUTRAL is a rejection class that lets the model say "a hand is visible but no sign is being made."

**In our stabilization pipeline:** A hand predicted as _NEUTRAL for 7 of the last 10 frames commits no word, ensuring silence between signs.

---

### Q15. What are your major misclassification pairs?

**Answer:** [TBD — fill from EXP-01 confusion matrix]

**Example framework answer:**  
"The confusion matrix showed [TBD] misclassifications between PLEASE and THANK_YOU. This is expected because both signs use a flat open palm with fingers together — they differ only in palm orientation (upward vs toward camera). This is a fine-grained distinction that requires the model to rely heavily on the relative z-coordinates of landmarks, which encode depth/orientation information."

---

## Category 4 — Evaluation

### Q16. Why macro F1 as the primary metric?

**Core answer:**  
Macro F1 averages the F1 score of each class equally, regardless of class size. Our 13 classes are all equally important — WATER is not less important than HELLO just because we collected fewer samples. Accuracy would give more weight to larger classes, potentially hiding poor performance on smaller classes. Macro F1 ensures the model performs well on ALL classes.

**Mathematical definition:**  
Macro F1 = (1/13) × Σ F1_c for c in all 13 classes

---

### Q17. How do you read a confusion matrix?

**Core answer:**  
The confusion matrix is a 13×13 grid:
- Rows = true class (what the sign actually was)
- Columns = predicted class (what the model predicted)
- Diagonal cells = correct predictions
- Off-diagonal cells = misclassifications

**Example:** If row THANK_YOU, column PLEASE = 8, then 8 actual THANK_YOU samples were incorrectly predicted as PLEASE.

**Your actual top confused pairs:** [TBD after EXP-01]

---

## Category 5 — System Performance and Limitations

### Q18. What are the limitations of your system?

**Core answer:**  
1. **Limited vocabulary:** Only 13 predefined gestures — cannot recognize arbitrary signs
2. **Static poses only:** Dynamic signs (involving motion) are captured as static representative poses
3. **Single signer:** Trained on one person's data — may not generalize to different hand sizes, skin tones, or signing styles
4. **Static background sensitivity:** The system works best in controlled lighting without extreme background clutter
5. **No grammar:** The system outputs individual sign labels — there is no sentence structure, grammar, or context modeling
6. **No temporal modeling:** Cannot distinguish between a held pose and an intentional sign of the same shape

---

### Q19. What happens if lighting changes significantly?

**Core answer:**  
MediaPipe's hand detection can fail in very dark conditions or with strong backlighting. During data collection, we deliberately varied lighting conditions across sessions to improve robustness. However, if the lighting at inference time is significantly outside the range seen during training, detection quality degrades, leading to more rejected frames and lower prediction confidence.

**Your actual observation:** [TBD after real-time testing]

---

### Q20. What happens if another person uses the model?

**Core answer:**  
The model was trained on a single signer's data. Different people have different hand shapes, sizes, and natural pose variations. The model may still work if the other person's signs fall within the feature distribution of the training data — but there is no guarantee. A truly generalizable model requires multi-signer training data. This is a known limitation of our single-signer dataset.

---

### Q21. How would you scale to more signs?

**Core answer:**  
The architecture supports adding more classes directly: collect data for new classes, re-train the classifier, and the pipeline handles any number of classes (limited only by feature discriminability). For a much larger vocabulary (hundreds of signs), the static landmark approach would need to be replaced with a temporal model (e.g., LSTM or Transformer) that captures motion sequences, as many signs are distinguished by their movement path rather than a static pose.

---

### Q22. Why does the system need stabilization?

**Core answer:**  
At 8 fps, the model produces 8 predictions per second. Without stabilization:
- A slight hand tremor during any sign would generate 8 text outputs per second
- One noisy frame could insert a wrong word
- Pauses between signs would generate continuous false predictions

The stabilization window (7 of 10 frames must agree) + confidence threshold (≥80%) + cooldown (1.2s) + neutral requirement together ensure that only deliberate, sustained, high-confidence signs are committed to the text output.

---

## Category 6 — Technical Depth

### Q23. How does the handedness fix work?

**Core answer:**  
MediaPipe assumes it receives a selfie/mirrored image (what you see in a front-facing phone camera). It labels handedness based on which side of the image the hand appears. Our webcam provides raw, un-mirrored frames. Without correction, physical right hand → appears on image-left → MediaPipe labels "Left" → wrong.

**Fix:** Before sending a frame to MediaPipe, we horizontally flip it (`cv2.flip(image, 1)`). MediaPipe now sees a mirrored frame and labels handedness correctly. After detection, we un-flip the x coordinates (`x = 1 - x`) so all downstream code works in the original camera frame. This is done transparently inside `HandDetector.detect_all()`.

---

### Q24. What is feature_set_version and why does it exist?

**Core answer:**  
`feature_set_version = "v1-base-63"` is a string saved in every Sample object and in `model_metadata.json`. When the inference pipeline loads a saved model, it checks that the current extractor's `feature_set_version` matches the model's recorded version. If they don't match — because the normalization code was changed between training and deployment — the system raises an error rather than silently producing wrong predictions. This guards against **training-serving skew**.

---

*Version 1.0 — 2026-09-23*  
*Update all [TBD] entries after running actual experiments.*
