

# import os
# import pandas as pd
# import librosa
# import numpy as np
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC

# # =========================
# # PATHS
# # =========================
# CSV_PATH = "dataset_full.csv"
# REAL_AUDIO_FOLDER = "/Users/Khushi/Desktop/ML/real audios"

# # =========================
# # FEATURE EXTRACTION
# # =========================
# def extract_mfcc_features(audio_path):
#     try:
#         audio, sr = librosa.load(audio_path, sr=16000)

#         audio = librosa.util.normalize(audio)
#         audio = audio[:16000 * 3]

#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
#         return None

#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     return np.mean(mfcc.T, axis=0)

# # =========================
# # LOAD DATASET (CSV + REAL FOLDER)
# # =========================
# def load_dataset():
#     X, y = [], []

#     # 🔹 1. Load CSV data
#     df = pd.read_csv(CSV_PATH)

#     for _, row in df.iterrows():
#         path = row["filepath"]
#         label = row["label"]

#         if os.path.exists(path):
#             features = extract_mfcc_features(path)
#             if features is not None:
#                 X.append(features)
#                 y.append(label)

#     print("Loaded CSV data:", len(X))

#     # 🔥 2. Load REAL AUDIO folder
#     real_count = 0

#     if os.path.exists(REAL_AUDIO_FOLDER):
#         for file in os.listdir(REAL_AUDIO_FOLDER):
#             if file.endswith(".flac") or file.endswith(".wav"):

#                 full_path = os.path.join(REAL_AUDIO_FOLDER, file)

#                 features = extract_mfcc_features(full_path)

#                 if features is not None:
#                     # 🔥 Add multiple times (boost impact)
#                     for _ in range(3):
#                         X.append(features)
#                         y.append(0)  # REAL

#                     real_count += 1

#     print("Loaded real audios:", real_count)

#     return np.array(X), np.array(y)

# # =========================
# # TRAIN MODEL
# # =========================
# def train():
#     print("Loading dataset...")
#     X, y = load_dataset()

#     print("Total data shape:", X.shape)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     model = SVC(kernel='rbf', C=1, probability=True)

#     print("Training model...")
#     model.fit(X_train, y_train)

#     acc = model.score(X_test, y_test)
#     print("Accuracy:", acc)

#     joblib.dump(model, "best_model.pkl")
#     joblib.dump(scaler, "scaler.pkl")

#     print("✅ Model saved!")

# # =========================
# if __name__ == "__main__":
#     train()


import os
import pandas as pd
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# MODELS
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# =========================
# PATHS
# =========================
CSV_PATH = "dataset_full.csv"
REAL_AUDIO_FOLDER = "/Users/Khushi/Desktop/ML/real audios"

# =========================
# FEATURE EXTRACTION (UNCHANGED)
# =========================
def extract_mfcc_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = librosa.util.normalize(audio)
        audio = audio[:16000 * 3]
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# =========================
# LOAD DATASET (UNCHANGED)
# =========================
def load_dataset():
    X, y = [], []

    df = pd.read_csv(CSV_PATH)

    for _, row in df.iterrows():
        path = row["filepath"]
        label = row["label"]

        if os.path.exists(path):
            features = extract_mfcc_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

    print("Loaded CSV data:", len(X))

    real_count = 0

    if os.path.exists(REAL_AUDIO_FOLDER):
        for file in os.listdir(REAL_AUDIO_FOLDER):
            if file.endswith(".flac") or file.endswith(".wav"):

                full_path = os.path.join(REAL_AUDIO_FOLDER, file)
                features = extract_mfcc_features(full_path)

                if features is not None:
                    for _ in range(3):
                        X.append(features)
                        y.append(0)

                    real_count += 1

    print("Loaded real audios:", real_count)

    return np.array(X), np.array(y)

# =========================
# TRAIN MODEL (UPDATED ONLY HERE)
# =========================
def train():
    print("Loading dataset...")
    X, y = load_dataset()

    print("Total data shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 🔥 MULTIPLE MODELS
    models = {
        "SVM": SVC(kernel='rbf', C=1, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3)
    }

    best_model = None
    best_score = 0
    best_name = ""

    print("\nTraining multiple models...\n")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    print("\n🏆 BEST MODEL:", best_name)
    print("Best Accuracy:", best_score)

    # SAVE BEST MODEL ONLY (same as before)
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("✅ Best model saved!")

# =========================
if __name__ == "__main__":
    train()