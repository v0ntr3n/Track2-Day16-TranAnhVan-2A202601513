import json
import time
from pathlib import Path

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# =========================
# CONFIG
# =========================

DATASET_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
OUTPUT_PATH = Path("benchmark_result.json")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main():
    # =========================
    # 1. LOAD DATASET
    # =========================

    print("Loading dataset...")

    start = time.perf_counter()
    df = pd.read_csv(DATASET_PATH)
    load_time = time.perf_counter() - start

    print(f"Dataset shape: {df.shape}")
    print(f"Load time: {load_time:.4f} seconds")

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Normal transactions: {(y == 0).sum()}")
    print(f"Fraud transactions:  {(y == 1).sum()}")

    # =========================
    # TRAIN / TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # =========================
    # 2. CREATE MODEL
    # =========================

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )

    # =========================
    # 3. TRAINING TIME
    # =========================

    print("\nTraining LightGBM...")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    print(f"Training time: {training_time:.4f} seconds")

    # =========================
    # 4. EVALUATION
    # =========================

    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print(f"AUC-ROC:   {auc_roc:.6f}")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"F1-Score:  {f1:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")

    # =========================
    # 5. INFERENCE BENCHMARK
    # =========================

    print("\nBenchmarking inference...")

    # warm-up
    model.predict_proba(X_test.iloc[:100])

    # ---- latency: 1 row ----

    single_row = X_test.iloc[[0]]

    runs = 100
    latencies = []

    for _ in range(runs):
        start = time.perf_counter()
        model.predict_proba(single_row)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    avg_latency_seconds = float(np.mean(latencies))
    avg_latency_ms = avg_latency_seconds * 1000

    # ---- throughput: 1000 rows ----

    batch_size = min(1000, len(X_test))
    batch = X_test.iloc[:batch_size]

    batch_runs = 10
    batch_times = []

    for _ in range(batch_runs):
        start = time.perf_counter()
        model.predict_proba(batch)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)

    avg_batch_time = float(np.mean(batch_times))

    throughput_rows_per_second = batch_size / avg_batch_time

    print(f"Average latency (1 row): {avg_latency_ms:.4f} ms")
    print(
        f"Throughput ({batch_size} rows): "
        f"{throughput_rows_per_second:.2f} rows/sec"
    )

    # =========================
    # 6. SAVE JSON
    # =========================

    best_iteration = getattr(model, "best_iteration_", None)

    results = {
        "dataset": {
            "path": str(DATASET_PATH),
            "total_rows": int(len(df)),
            "features": int(X.shape[1]),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "fraud_rows": int((y == 1).sum()),
            "normal_rows": int((y == 0).sum()),
        },
        "timing": {
            "load_data_seconds": round(load_time, 6),
            "training_seconds": round(training_time, 6),
        },
        "metrics": {
            "auc_roc": round(float(auc_roc), 6),
            "accuracy": round(float(accuracy), 6),
            "f1_score": round(float(f1), 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
        },
        "inference": {
            "single_row_latency_ms": round(avg_latency_ms, 6),
            "single_row_runs": runs,
            "batch_size": batch_size,
            "batch_average_seconds": round(avg_batch_time, 6),
            "throughput_rows_per_second": round(
                throughput_rows_per_second, 2
            ),
        },
        "model": {
            "type": "LGBMClassifier",
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "num_leaves": model.num_leaves,
            "best_iteration": best_iteration,
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH.resolve()}")

    # =========================
    # SUMMARY
    # =========================

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"Load data time       : {load_time:.4f} s")
    print(f"Training time        : {training_time:.4f} s")
    print(f"Best iteration       : {best_iteration}")
    print(f"AUC-ROC              : {auc_roc:.6f}")
    print(f"Accuracy             : {accuracy:.6f}")
    print(f"F1-Score             : {f1:.6f}")
    print(f"Precision            : {precision:.6f}")
    print(f"Recall               : {recall:.6f}")
    print(f"Inference latency    : {avg_latency_ms:.4f} ms / row")
    print(
        f"Inference throughput : "
        f"{throughput_rows_per_second:.2f} rows/sec"
    )


if __name__ == "__main__":
    main()
