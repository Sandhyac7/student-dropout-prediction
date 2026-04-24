import os
import numpy as np
import joblib
from flask import Flask, render_template, request

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask setup (force correct paths)
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Load models safely
models = {}
model_files = {
    "Logistic Regression": "lr.pkl",
    "Random Forest": "rf.pkl",
    "SVM": "svm.pkl",
    "XGBoost": "xgb.pkl"
}

for name, file in model_files.items():
    path = os.path.join(BASE_DIR, "models", file)
    if os.path.exists(path):
        models[name] = joblib.load(path)
    else:
        print(f"⚠️ Model not found: {file}")

# Load scaler
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        attendance = float(request.form["attendance"])
        marks = float(request.form["marks"])
        study_hours = float(request.form["study_hours"])

        # Prepare input
        features = np.array([[attendance, marks, study_hours]])

        # Apply scaling if available
        if scaler:
            features = scaler.transform(features)

        # Predict using all models
        results = {}
        for name, model in models.items():
            prob = model.predict_proba(features)[0][1]
            results[name] = round(prob * 100, 2)

        return render_template("index.html", predictions=results)

    except Exception as e:
        return f"Error: {str(e)}"


# Debug prints
if __name__ == "__main__":
    print("📁 Running from:", BASE_DIR)
    
    templates_path = os.path.join(BASE_DIR, "templates")
    print("📂 Templates path:", templates_path)
    print("📄 Template files:", os.listdir(templates_path) if os.path.exists(templates_path) else "Missing")

    models_path = os.path.join(BASE_DIR, "models")
    print("🧠 Models path:", models_path)
    print("📦 Model files:", os.listdir(models_path) if os.path.exists(models_path) else "Missing")

    app.run(debug=True)