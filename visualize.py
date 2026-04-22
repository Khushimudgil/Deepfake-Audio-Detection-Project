import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.svm import SVC

# FEATURE EXTRACTION
def extract_mfcc_features(audio_path):
    try:
        audio_data, sr = librosa.load(audio_path, sr=16000)
    except:
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# LOAD DATASET
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    X, y = [], []

    for _, row in df.iterrows():
        path = row["filepath"]
        label = row["label"]

        if os.path.exists(path):
            mfcc = extract_mfcc_features(path)
            if mfcc is not None:
                X.append(mfcc)
                y.append(label)

    return np.array(X), np.array(y)

# TRAIN + VISUALIZE
def visualize_model(csv_path):
    print("Loading dataset...")
    X, y = load_dataset(csv_path)

    print("Dataset shape:", X.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVM
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 2. ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # 3. ACTUAL vs PREDICTED 
    plt.figure()
    plt.scatter(y_test, y_pred)

    # Ideal line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.title("Actual vs Predicted")
    plt.show()

    # 4. CLASS DISTRIBUTION
    plt.figure()
    sns.countplot(x=y)
    plt.title("Class Distribution (0 = Real, 1 = Fake)")
    plt.show()

    # 5. MFCC FEATURE DISTRIBUTION
    plt.figure()
    plt.plot(X[0])
    plt.title("Sample MFCC Feature Vector")
    plt.xlabel("MFCC Coefficient Index")
    plt.ylabel("Value")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# RUN
if __name__ == "__main__":
    csv_path = "/Users/Khushi/Desktop/ML/DeepFake-Audio-Detection-MFCC/dataset_full.csv"
    visualize_model(csv_path)
