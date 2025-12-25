#!/usr/bin/env python
# coding: utf-8

"""
Emotion Detection in Text
Training and evaluation pipeline using Logistic Regression.
Metrics and artifacts are logged to MLflow.
"""

# =========================
# Imports
# =========================
from pathlib import Path
import os
import pandas as pd
import numpy as np
import neattext.functions as nfx
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

import joblib
import mlflow
import mlflow.sklearn
import json


# =========================
# MLflow configuration
# =========================
MLFLOW_EXPERIMENT_NAME = "emotion_detection_experiment"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# =========================
# Paths (ROBUSTE CI)
# =========================
# application/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "cleaned_dataset.csv"
MODEL_PATH = BASE_DIR / "src" / "models" / "emotion_classifier_pipe_lr.pkl"
TEMP_METRICS_PATH = BASE_DIR / "metrics_new.json"

print(f"[INFO] Loading dataset from: {DATA_PATH}")


# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH, encoding="utf-8")

# =========================
# Data exploration
# =========================
sns.countplot(x="sentiment", data=df)
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
mlflow.log_artifact("sentiment_distribution.png")
plt.close()


# =========================
# Data cleaning
# =========================
df["text"] = df["text"].fillna("").astype(str)
df["clean_text"] = (
    df["text"]
    .apply(nfx.remove_userhandles)
    .apply(nfx.remove_stopwords)
)

X = df["clean_text"]
y = df["sentiment"]


# =========================
# Train / test split
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# =========================
# Pipeline
# =========================
pipe_lr = Pipeline(
    steps=[
        ("cv", CountVectorizer()),
        ("lr", LogisticRegression(max_iter=100, solver="lbfgs")),
    ]
)


# =========================
# MLflow run
# =========================
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run():
    # ---- Train
    pipe_lr.fit(x_train, y_train)

    # ---- Params
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("vectorizer", "CountVectorizer")
    mlflow.log_param("max_iter", pipe_lr["lr"].max_iter)
    mlflow.log_param("solver", pipe_lr["lr"].solver)
    mlflow.log_param("train_test_split", "70/30")
    mlflow.log_param("vocab_size", len(pipe_lr["cv"].vocabulary_))

    # ---- Evaluation
    preds = pipe_lr.predict(x_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="weighted")
    recall = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    with open(TEMP_METRICS_PATH, "w") as f:
     json.dump(metrics, f, indent=2)

    mlflow.log_artifact(METRICS_PATH)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ---- Confusion matrix
    cm = ConfusionMatrixDisplay.from_estimator(pipe_lr, x_test, y_test)
    cm.figure_.tight_layout()
    cm.figure_.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # ---- Save model (ONE source of truth)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_lr, MODEL_PATH)

    mlflow.log_artifact(MODEL_PATH)

    # ---- Log dataset snapshot
    df.to_csv("cleaned_dataset_snapshot.csv", index=False)
    mlflow.log_artifact("cleaned_dataset_snapshot.csv")

    print(f"[SUCCESS] Model trained | accuracy={accuracy:.4f}")


# =========================
# Test prediction
# =========================
test_text = "This book was so interesting it made me happy"
prediction = pipe_lr.predict([test_text])[0]
confidence = np.max(pipe_lr.predict_proba([test_text]))

print(f"[TEST] Text: {test_text}")
print(f"[TEST] Prediction: {prediction} (confidence={confidence:.2f})")
