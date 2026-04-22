
import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib

app = Flask(__name__)

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

LABEL_MAP = {
    0: "REAL AUDIO",
    1: "DEEPFAKE AUDIO"
}

# FEATURE EXTRACTION
def extract_mfcc_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        audio = librosa.util.normalize(audio)

        audio = audio[:16000 * 3]

    except Exception as e:
        print("Error processing:", e)
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# ANALYZE AUDIO
def analyze_audio(path):

    if not os.path.exists(path):
        return "File not found"

    features = extract_mfcc_features(path)

    if features is None:
        return "Could not process audio"

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    features = scaler.transform(features.reshape(1, -1))

    proba = model.predict_proba(features)[0]

    real_prob = proba[0]
    fake_prob = proba[1]

    print("Real:", real_prob, "Fake:", fake_prob) 

    # DEMO SAFE LOGIC
    if abs(real_prob - fake_prob) < 0.15:
        return f"⚠️ UNCERTAIN RESULT (Real: {real_prob:.2f}, Fake: {fake_prob:.2f})"

    if real_prob > fake_prob:
        return f"✅ REAL AUDIO (Confidence: {real_prob:.2f})"
    else:
        return f"❌ DEEPFAKE AUDIO (Confidence: {fake_prob:.2f})"

# ROUTES
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get("audio_file")

        if not file or file.filename == "":
            return render_template("index.html", message="No file uploaded")

        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        path = os.path.join("uploads", file.filename)
        file.save(path)

        result = analyze_audio(path)

        os.remove(path)

        return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
