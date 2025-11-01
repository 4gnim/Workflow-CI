import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os


def main():
    print("Memulai skrip pelatihan 'modelling.py'...")

    # Muat data
    try:
        df = pd.read_csv("clean_credit_card_fraud.csv")
    except FileNotFoundError:
        print("Error: File 'clean_credit_card_fraud.csv' tidak ditemukan.")
        return

    # Pisahkan fitur dan target
    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # parameter default
    n_estimators = 100
    max_depth = 10

    # log parameter
    print("Logging parameters...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Latih model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi
    print("Evaluating model...")
    y_pred = model.predict(X_test)

    # log Metrik
    print("Logging metrics...")
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))

    # log Model
    print("Logging model...")
    # 'model' adalah nama folder artefak yang akan dicari oleh build-docker
    mlflow.sklearn.log_model(model, "model")

    print("Pelatihan selesai. Log otomatis masuk ke run yang aktif.")


if __name__ == "__main__":
    main()
