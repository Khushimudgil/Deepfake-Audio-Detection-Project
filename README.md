# 🎧 Deepfake Audio Detection using Machine Learning

## Overview

This project focuses on detecting **deepfake (synthetic) audio** using machine learning techniques. It leverages **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction and evaluates multiple classification models to distinguish between **real and fake audio samples**.

The final system is deployed using **Flask**, allowing users to upload an audio file and get real-time predictions.

## Features

* Audio feature extraction using MFCC
* Multiple ML models implemented:

  * Support Vector Machine (SVM)
  * Random Forest
  * Logistic Regression
  * Gradient Boosting
*  Model comparison using:

  * Accuracy
  * Precision
  * F1 Score
*  Automatic best model selection
*  Flask-based web interface for real-time prediction
*  Supports both `.wav` and `.flac` audio formats

## Tech Stack

* Python
* NumPy, Pandas
* Librosa (Audio Processing)
* Scikit-learn (ML Models)
* Flask (Deployment)
* Joblib (Model Saving)

## Project Structure

```
ML/
 └── DeepFake-Audio-Detection-MFCC/
      ├── app.py                # Flask app
      ├── main.py                # Training script
      ├── best_model.pkl        # Saved model
      ├── scaler.pkl            # Feature scaler
      ├── index.html           # HTML files
      ├── visualize.py             # Visual Aid
      └── README.md
```

---

## Dataset

This project uses the **ASVspoof 2019 dataset**, a widely used benchmark for **audio deepfake detection and spoofing attacks**.

The dataset includes:

* Genuine (real human speech)
* Spoofed audio generated using:

  * Speech synthesis
  * Voice conversion
  * Replay attacks ([arXiv][1])

It is divided into:

* Training set
* Development set
* Evaluation set (includes unseen attacks for generalization) ([Kaggle][2])

**Note:**
The dataset is **very large in size**, so it is **not included in this repository**.

Download it from Kaggle:
https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset?utm_source=chatgpt.com

After downloading, update the dataset paths in your code accordingly.

---

## How It Works

1. Extract MFCC features from audio files
2. Convert audio into numerical feature vectors
3. Train multiple ML models
4. Evaluate performance using standard metrics
5. Select the best-performing model
6. Deploy using Flask for real-time predictions

---

## How to Run

### Step 1: Install Dependencies

```
pip install flask librosa scikit-learn numpy pandas joblib
```

---

### Step 2: Train Model

```
python new.py
```

This will generate:

* `best_model.pkl`
* `scaler.pkl`

---

### Step 3: Run Web App

```
python app.py
```

---

### Step 4: Open in Browser

```
http://127.0.0.1:5000
```

Upload an audio file to check whether it is **real or deepfake**.

---

## Results

* Achieved **~97% accuracy** using SVM
* High precision and F1-score on imbalanced dataset
* Robust performance across multiple models

---

## Authors

Khushi Mudgil 
Shristi
Daksh Bansal

[1]: https://arxiv.org/abs/1911.01601?utm_source=chatgpt.com "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech"
[2]: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset?utm_source=chatgpt.com "ASVspoof 2019 Dataset"
