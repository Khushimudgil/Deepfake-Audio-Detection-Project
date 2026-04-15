
# import os
# from flask import Flask, request, render_template
# import librosa
# import numpy as np
# import joblib

# app = Flask(__name__)

# # =========================
# # FEATURE EXTRACTION (MATCH TRAINING)
# # =========================
# def extract_mfcc_features(audio_path):
#     try:
#         audio_data, sr = librosa.load(audio_path, sr=16000)  # ✅ same as training
#     except Exception as e:
#         print(f"Error loading audio file {audio_path}: {e}")
#         return None

#     mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
#     return np.mean(mfccs.T, axis=0)


# # =========================
# # ANALYZE AUDIO
# # =========================
# def analyze_audio(input_audio_path):
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_filename = os.path.join(BASE_DIR, "best_model.pkl")
#     scaler_filename = os.path.join(BASE_DIR, "scaler.pkl")

#     if not os.path.exists(input_audio_path):
#         return "Error: File does not exist."

#     if not (input_audio_path.lower().endswith(".wav") or input_audio_path.lower().endswith(".flac")):
#         return "Error: Only .wav or .flac files allowed."

#     mfcc_features = extract_mfcc_features(input_audio_path)

#     if mfcc_features is not None:
#         scaler = joblib.load(scaler_filename)
#         model = joblib.load(model_filename)

#         mfcc_scaled = scaler.transform(mfcc_features.reshape(1, -1))

#         prediction = model.predict(mfcc_scaled)
#         proba = model.predict_proba(mfcc_scaled)  # 🔥 confidence

#         confidence = np.max(proba)

#         if prediction[0] == 0:
#             return f"✅ REAL AUDIO (Confidence: {confidence:.2f})"
#         else:
#             return f"❌ DEEPFAKE AUDIO (Confidence: {confidence:.2f})"

#     return "Error: Could not process audio."


# # =========================
# # ROUTES
# # =========================
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "audio_file" not in request.files:
#             return render_template("index.html", message="No file uploaded")

#         audio_file = request.files["audio_file"]

#         if audio_file.filename == "":
#             return render_template("index.html", message="No file selected")

#         if audio_file and allowed_file(audio_file.filename):
#             if not os.path.exists("uploads"):
#                 os.makedirs("uploads")

#             audio_path = os.path.join("uploads", audio_file.filename)
#             audio_file.save(audio_path)

#             result = analyze_audio(audio_path)

#             os.remove(audio_path)

#             return render_template("result.html", result=result)

#         return render_template("index.html", message="Only .wav or .flac allowed")

#     return render_template("index.html")


# # =========================
# # FILE VALIDATION
# # =========================
# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ["wav", "flac"]


# # =========================
# # RUN APP
# # =========================
# if __name__ == "__main__":
#     app.run(debug=True)


# import os
# from flask import Flask, request, render_template
# import librosa
# import numpy as np
# import joblib

# app = Flask(__name__)

# # =========================
# # CONFIG
# # =========================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
# SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# LABEL_MAP = {
#     0: "REAL AUDIO",
#     1: "DEEPFAKE AUDIO"
# }

# # =========================
# # FEATURE EXTRACTION
# # =========================
# def extract_mfcc_features(audio_path):
#     try:
#         audio, sr = librosa.load(audio_path, sr=16000)

#         audio = librosa.util.normalize(audio)
#         audio = audio[:16000 * 3]

#     except:
#         return None

#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return np.mean(mfcc.T, axis=0)

# # =========================
# # ANALYZE AUDIO
# # =========================
# def analyze_audio(path):

#     if not os.path.exists(path):
#         return "❌ File not found"

#     features = extract_mfcc_features(path)

#     if features is None:
#         return "❌ Could not process audio"

#     scaler = joblib.load(SCALER_PATH)
#     model = joblib.load(MODEL_PATH)

#     features = scaler.transform(features.reshape(1, -1))

#     pred = model.predict(features)
#     proba = model.predict_proba(features)

#     confidence = float(np.max(proba))

#     # 🔥 DEMO SAFE
#     if confidence < 0.75:
#         return f"⚠️ UNCERTAIN RESULT (Confidence: {confidence:.2f})"

#     label = LABEL_MAP[pred[0]]

#     if label == "REAL AUDIO":
#         return f"✅ {label} (Confidence: {confidence:.2f})"
#     else:
#         return f"❌ {label} (Confidence: {confidence:.2f})"

# # =========================
# # ROUTES
# # =========================
# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file = request.files.get("audio_file")

#         if not file or file.filename == "":
#             return render_template("index.html", message="No file uploaded")

#         if not os.path.exists("uploads"):
#             os.makedirs("uploads")

#         path = os.path.join("uploads", file.filename)
#         file.save(path)

#         result = analyze_audio(path)

#         os.remove(path)

#         return render_template("result.html", result=result)

#     return render_template("index.html")

# # =========================
# if __name__ == "__main__":
#     app.run(debug=True, port=5001)


import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import joblib

app = Flask(__name__)

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

LABEL_MAP = {
    0: "REAL AUDIO",
    1: "DEEPFAKE AUDIO"
}

# =========================
# FEATURE EXTRACTION
# =========================
def extract_mfcc_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)

        # Normalize audio
        audio = librosa.util.normalize(audio)

        # Fix length (3 sec)
        audio = audio[:16000 * 3]

    except Exception as e:
        print("Error processing:", e)
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# =========================
# ANALYZE AUDIO
# =========================
def analyze_audio(path):

    if not os.path.exists(path):
        return "❌ File not found"

    features = extract_mfcc_features(path)

    if features is None:
        return "❌ Could not process audio"

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    features = scaler.transform(features.reshape(1, -1))

    proba = model.predict_proba(features)[0]

    real_prob = proba[0]
    fake_prob = proba[1]

    print("Real:", real_prob, "Fake:", fake_prob)  # debug (optional)

    # 🔥 DEMO SAFE LOGIC
    if abs(real_prob - fake_prob) < 0.15:
        return f"⚠️ UNCERTAIN RESULT (Real: {real_prob:.2f}, Fake: {fake_prob:.2f})"

    if real_prob > fake_prob:
        return f"✅ REAL AUDIO (Confidence: {real_prob:.2f})"
    else:
        return f"❌ DEEPFAKE AUDIO (Confidence: {fake_prob:.2f})"

# =========================
# ROUTES
# =========================
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

# =========================
if __name__ == "__main__":
    app.run(debug=True, port=5001)