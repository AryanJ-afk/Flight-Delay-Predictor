# Flight-Delay-Predictor
A machine learning model on the January Flight Delay dataset, using flight schedule, route, carrier, and distance features to classify whether a flight would be delayed.


## Impact of Class Weighting on Flight Delay Prediction

When predicting **flight delays**, the dataset is heavily imbalanced: most flights are on time, and only a minority are delayed.
This imbalance caused the initial models to favor the majority class (on-time flights), achieving high accuracy but very poor recall for delays.

To address this, we applied **class balancing techniques**:

* `class_weight="balanced"` for Logistic Regression and Random Forest
* `scale_pos_weight` for XGBoost

The results highlight how weighting changes model behavior:

---

### **1. Logistic Regression**

* **Without weights**:

  * Very high accuracy (\~0.89–0.90)
  * Very low recall for delays (\~0.23–0.28) → barely flagged delayed flights
* **With weights**:

  * Accuracy dropped (\~0.72–0.75)
  * Recall jumped dramatically (\~0.67–0.72) → model actually flagged most delays
  * Tradeoff: more false positives, but much more useful in practice

**Takeaway:** Class weighting transformed logistic regression from an "always on-time" predictor into a usable baseline model for detecting delays.

---

### **2. Random Forest**

* **Without weights**:

  * High accuracy (\~0.86)
  * Recall for delays nearly zero (\~0.00–0.01) → useless for minority class
* **With weights**:

  * Accuracy dropped (\~0.70–0.73)
  * Recall surged (\~0.68–0.76)
  * Precision decreased, but F1-score improved for delayed flights

**Takeaway:** Random Forest needed class weighting to even recognize delays. With weights, it became far more balanced, though less precise than XGBoost.

---

### **3. XGBoost**

* **Without weights**:

  * Already strong performance out-of-the-box
  * Accuracy \~0.93–0.94, Recall \~0.51–0.61, ROC AUC \~0.92–0.94
* **With `scale_pos_weight`**:

  * Accuracy slightly reduced (\~0.89–0.91)
  * Recall jumped significantly (\~0.78 for both arrival and departure delays)
  * ROC AUC remained excellent (\~0.92–0.94)
  * Precision held steady (\~0.56–0.64), keeping F1 balanced

**Takeaway:** XGBoost performed best overall. With proper weighting, it achieved the highest recall **without sacrificing too much accuracy**, making it the most reliable model for predicting flight delays.

---
