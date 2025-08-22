# Flight-Delay-Predictor
A machine learning model on the January Flight Delay dataset, using flight schedule, route, carrier, and distance features to classify whether a flight would be delayed.

## Dataset Link: https://www.kaggle.com/datasets/divyansh22/flight-delay-prediction?select=Jan_2020_ontime.csv

## Model Performance Progression

We trained Logistic Regression, Random Forest, and XGBoost models to predict **arrival and departure delays (≥15 minutes)**.

### 1. Baseline (no class weights, raw HHMM times)

* **Logistic Regression:**

  * Accuracy \~89–90% but very poor recall for delays (\~0.23–0.28).
  * Strong bias toward the majority “on-time” class.
  * ROC AUC: \~0.83–0.89.

* **Random Forest:**

  * Accuracy \~86%.
  * Detected almost no delays (recall \~0.005).
  * ROC AUC: \~0.82–0.86.

* **XGBoost:**

  * Best baseline.
  * Accuracy \~92–94%, recall for delays \~0.51–0.61.
  * ROC AUC: \~0.92–0.94.

---

### 2.With Class Weights

Balancing the minority class (delays) improved recall significantly but lowered accuracy.

* **Logistic Regression:**

  * Recall jumped to \~0.67–0.72, accuracy dropped to \~72–75%.
  * ROC AUC: \~0.77–0.81.

* **Random Forest:**

  * Recall \~0.68–0.76, accuracy \~70–73%.
  * ROC AUC: \~0.79–0.81.

* **XGBoost (with `scale_pos_weight`):**

  * Recall \~0.77–0.78 while maintaining strong accuracy (\~89–91%).
  * ROC AUC stayed very high (\~0.92–0.94).

---

### 3. With Class Weights + Cyclical Time Encoding (sin/cos for DEP\_TIME & ARR\_TIME)

Replacing raw HHMM times with **sine/cosine encodings** gave models a better sense of the 24-hour cycle.

* **Logistic Regression:**

  * Accuracy \~71–74%, recall \~0.64–0.69.
  * ROC AUC: \~0.74–0.78.
  * Much stronger at identifying delays than baseline.

* **Random Forest:**

  * Accuracy \~76–77%, recall \~0.72–0.79.
  * ROC AUC: \~0.83–0.86.
  * Significant gains in recall without too much accuracy loss.

* **XGBoost:**

  * Accuracy \~89–91%, recall \~0.77–0.79.
  * ROC AUC: \~0.92–0.94.
  * Maintained strong overall performance and best balance between precision & recall.

---

### Key Insights

* **Baseline:** All models were biased toward “on-time,” missing many delays.
* **Class Weights:** Improved recall, especially for Logistic Regression and Random Forest, though at the cost of accuracy.
* **Class Weights + Cyclical Encoding:** Further boosted recall and overall balance, particularly for Logistic Regression and Random Forest, while XGBoost remained the best all-around model.
