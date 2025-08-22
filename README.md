# Flight-Delay-Predictor
A machine learning model on the January Flight Delay dataset, using flight schedule, route, carrier, and distance features to classify whether a flight would be delayed.

Logistic Regression establishes a strong baseline with ROC AUC of 0.83 (arrival) and 0.89 (departure). Departure delays are easier to model since they depend on schedule and airport factors available before takeoff, whereas arrival delays reflect additional en-route uncertainties. However, recall for delayed flights is low, highlighting the difficulty of catching true delays.
