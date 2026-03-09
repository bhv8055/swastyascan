import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv(r"C:\Users\bhara\Downloads\Swastyascan\datasets\heart\heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost model
model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100,200,300],
    "max_depth": [3,5,7],
    "learning_rate": [0.01,0.05,0.1],
    "subsample": [0.8,1],
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Predictions
pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save model
joblib.dump(best_model,
r"C:\Users\bhara\Downloads\Swastyascan\models\heart_model.pkl")

# Save scaler
joblib.dump(scaler,
r"C:\Users\bhara\Downloads\Swastyascan\models\heart_scaler.pkl")

print("Heart model trained successfully")