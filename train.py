import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# Dataset
data = pd.DataFrame({
    'attendance': [80, 60, 30, 90, 50, 20, 85, 40],
    'marks': [70, 50, 30, 85, 45, 25, 90, 35],
    'study_hours': [3, 2, 1, 4, 2, 1, 5, 2],
    'dropout': [0, 1, 1, 0, 1, 1, 0, 1]
})

X = data[['attendance', 'marks', 'study_hours']]
y = data['dropout']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "lr": LogisticRegression(),
    "rf": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "xgb": XGBClassifier(eval_metric='logloss')
}

# Train & save
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")

# Save scaler too (IMPORTANT for Flask)
joblib.dump(scaler, "models/scaler.pkl")

print("Models trained and saved!")