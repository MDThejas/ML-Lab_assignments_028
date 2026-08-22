import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# =====================================================
# IMPORT YOUR IMPLEMENTATION
# =====================================================

from my_code import (
    fit as my_fit,
    predict as my_predict
)

# =====================================================
# IMPORT AI IMPLEMENTATION
# =====================================================

from Ai_written_code import (
    fit as ai_fit,
    predict as ai_predict
)

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_excel(
    "Lab Session Data.xlsx",
    sheet_name="marketing_campaign"
)

# =====================================================
# SIMPLE PREPROCESSING
# =====================================================

for col in data.columns:

    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(
            data[col].median()
        )

    else:
        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

for col in data.select_dtypes(
    include=["object"]
).columns:

    data[col] = pd.factorize(
        data[col]
    )[0]

if "Response" not in data.columns:
    raise Exception(
        "Response column not found"
    )

X = data.drop(
    "Response",
    axis=1
).values

y = data["Response"].values

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================================================
# METRIC FUNCTION
# =====================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "Accuracy":
        accuracy_score(
            y_true,
            y_pred
        ),

        "Precision":
        precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall":
        recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1":
        f1_score(
            y_true,
            y_pred,
            zero_division=0
        )
    }


# =====================================================
# MY KNN
# =====================================================

def evaluate_my_knn(k):

    value = my_fit(
        X_train,
        y_train
    )

    start = time.perf_counter()

    for _ in range(10):

        y_pred = []

        for row in X_test:

            y_pred.append(
                my_predict(
                    value,
                    row,
                    k
                )
            )

    end = time.perf_counter()

    avg_time = (
        end - start
    ) / 10

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    metrics["Time"] = avg_time

    return metrics


# =====================================================
# AI KNN
# =====================================================

def evaluate_ai_knn(k):

    ai_fit(
        X_train,
        y_train
    )

    start = time.perf_counter()

    for _ in range(10):

        y_pred = []

        for row in X_test:

            y_pred.append(
                ai_predict(
                    "merge",
                    row,
                    k
                )
            )

    end = time.perf_counter()

    avg_time = (
        end - start
    ) / 10

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    metrics["Time"] = avg_time

    return metrics


# =====================================================
# SKLEARN KNN
# =====================================================

def evaluate_sklearn_knn(k):

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    start = time.perf_counter()

    for _ in range(10):

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

    end = time.perf_counter()

    avg_time = (
        end - start
    ) / 10

    metrics = calculate_metrics(
        y_test,
        y_pred
    )

    metrics["Time"] = avg_time

    return metrics


# =====================================================
# RUN COMPARISON
# =====================================================

k = 5

my_result = evaluate_my_knn(k)

ai_result = evaluate_ai_knn(k)

sk_result = evaluate_sklearn_knn(k)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Avg Time (s)"
    ],

    "My kNN": [

        my_result["Accuracy"],
        my_result["Precision"],
        my_result["Recall"],
        my_result["F1"],
        my_result["Time"]
    ],

    "AI kNN": [

        ai_result["Accuracy"],
        ai_result["Precision"],
        ai_result["Recall"],
        ai_result["F1"],
        ai_result["Time"]
    ],

    "Sklearn kNN": [

        sk_result["Accuracy"],
        sk_result["Precision"],
        sk_result["Recall"],
        sk_result["F1"],
        sk_result["Time"]
    ]
})

print("\n")
print(comparison)

# =====================================================
# PLOT
# =====================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

my_values = [
    my_result["Accuracy"],
    my_result["Precision"],
    my_result["Recall"],
    my_result["F1"]
]

ai_values = [
    ai_result["Accuracy"],
    ai_result["Precision"],
    ai_result["Recall"],
    ai_result["F1"]
]

sk_values = [
    sk_result["Accuracy"],
    sk_result["Precision"],
    sk_result["Recall"],
    sk_result["F1"]
]

x = np.arange(
    len(metrics)
)

width = 0.25

plt.figure(
    figsize=(10,6)
)

plt.bar(
    x-width,
    my_values,
    width,
    label="My kNN"
)

plt.bar(
    x,
    ai_values,
    width,
    label="AI kNN"
)

plt.bar(
    x+width,
    sk_values,
    width,
    label="Sklearn kNN"
)

plt.xticks(
    x,
    metrics
)

plt.ylabel(
    "Score"
)

plt.title(
    "kNN Performance Comparison"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.show()