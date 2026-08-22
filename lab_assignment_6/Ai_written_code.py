print("A1\n")
import pandas as pd
import numpy as np
import unittest
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


#Chatgpt
print("A1\n")
# -----------------------------
# Distance Function
# -----------------------------
def eu_distance(a, b):
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    return np.sqrt(np.sum(diff ** 2))


# -----------------------------
# Missing Value Handling
# -----------------------------
def fill_missing(data):
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())
    return data


# -----------------------------
# Encoding Functions
# -----------------------------
def label_encoding(column):
    unique_vals = column.unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    return column.map(mapping)


def one_hot_encoding(column):
    return pd.get_dummies(column, prefix=column.name)


# -----------------------------
# Bubble Sort
# -----------------------------
def bubble_sort(arr):
    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j][0] > arr[j + 1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True

        if not swapped:
            break

    return arr


# -----------------------------
# Insertion Sort
# -----------------------------
def insertion_sort(arr):

    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j][0] > key[0]:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


# -----------------------------
# Merge Sort
# -----------------------------
def merge_sort(arr):

    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2

    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    merged = []
    i = j = 0

    while i < len(left) and j < len(right):

        if left[i][0] < right[j][0]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


# -----------------------------
# Majority Voting
# -----------------------------
def most_number_class(arr):
    freq = Counter(arr)
    return freq.most_common(1)[0][0]


# -----------------------------
# Neighbour Selection
# -----------------------------
def get_neighbours(X_train, y_train, test_row, k, sort_type):

    distances = []

    for i in range(len(X_train)):
        dist = eu_distance(X_train[i], test_row)
        distances.append((dist, y_train[i]))

    if sort_type == "bubble":
        distances = bubble_sort(distances)

    elif sort_type == "insertion":
        distances = insertion_sort(distances)

    elif sort_type == "merge":
        distances = merge_sort(distances)

    neighbours = [label for _, label in distances[:k]]

    return neighbours


# -----------------------------
# Prediction Function
# -----------------------------
def predict_knn(X_train, y_train, X_test, k=5, sort_type="merge"):

    predictions = []

    for row in X_test:

        neighbours = get_neighbours(
            X_train,
            y_train,
            row,
            k,
            sort_type
        )

        pred = most_number_class(neighbours)
        predictions.append(pred)

    return np.array(predictions)


# ==================================================
# Load Marketing Campaign Dataset
# ==================================================

data = pd.read_excel(
    "Lab Session Data.xlsx",
    sheet_name="marketing_campaign"
)

data = fill_missing(data)

# Label Encoding
for col in ["Education", "Marital_Status"]:
    data[col] = label_encoding(data[col])

# Drop date column
data = data.drop("Dt_Customer", axis=1)

# Features and Target
X = data.drop("Response", axis=1)
y = data["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.2,
    random_state=42
)

# Run kNN
predictions = predict_knn(
    X_train,
    y_train,
    X_test,
    k=5,
    sort_type="merge"     # bubble / insertion / merge
)

acc = accuracy_score(y_test, predictions)

print("Accuracy :", round(acc * 100, 2), "%")
print("\n")

#Chatgpt
print("A2\n")
# -----------------------------
# Weighted Voting
# -----------------------------
def weighted_class(arr):

    weights = {}

    for dist, label in arr:

        weight = 1 / (dist + 1e-9)

        if label not in weights:
            weights[label] = 0

        weights[label] += weight

    return max(weights, key=weights.get)


# -----------------------------
# Weighted Prediction
# -----------------------------
def w_predict(value, test_row, k):

    distances = []

    for i in range(len(X_train)):

        dist = eu_distance(X_train[i], test_row)

        distances.append(
            (dist, y_train[i])
        )

    if value == "bubble":
        distances = bubble_sort(distances)

    elif value == "insertion":
        distances = insertion_sort(distances)

    elif value == "merge":
        distances = merge_sort(distances)

    neighbours = distances[:k]

    return weighted_class(neighbours)
predictions = []

for row in X_test:

    pred = w_predict(
        "merge",
        row,
        k=5
    )

    predictions.append(pred)

predictions = np.array(predictions)
print("\n")


#Chatgpt
print("A3\n")

# Features and Target
X = data.drop("Response", axis=1).values
y = data["Response"].values

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Total Records :", len(X))
print("Training Records :", len(X_train))
print("Testing Records :", len(X_test))

print("\nX_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)
print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)
print("\n")

#Chatgpt
print("A4\n")

# Create kNN model
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the model
neigh.fit(X_train, y_train)
print("kNN model trained successfully!")
print("\n")

#Chatgpt
print("A5\n")
# Train kNN Classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Evaluate Accuracy
accuracy = neigh.score(X_test, y_test)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("\n")

#Chatgpt
print("A6\n")

# Train kNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Evaluate
accuracy = neigh.score(X_test, y_test)

print("k Value :", 3)
print("Test Accuracy :", round(accuracy * 100, 2), "%")
print("\n")


#Chatgpt
print("A7\n")
# ==========================
# CUSTOM KNN CLASSIFIER
# ==========================
stored_X = None
stored_y = None


def fit(x_train, y_train):

    global stored_X, stored_y

    stored_X = np.asarray(x_train, dtype=float)
    stored_y = np.asarray(y_train)

    return {
        "samples": stored_X.shape[0],
        "features": stored_X.shape[1]
    }


def predict(value, test_row, k):

    if stored_X is None:
        raise Exception("Model has not been fitted")

    neighbours = get_neighbours(
        stored_X,
        stored_y,
        np.asarray(test_row, dtype=float),
        k,
        value
    )

    prediction = most_number_class(neighbours)

    return prediction


def score(value, X_test, y_test, k):

    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)

    predicted_labels = [
        predict(value, row, k)
        for row in X_test
    ]

    comparison = np.equal(
        predicted_labels,
        y_test
    )

    accuracy = (
        np.count_nonzero(comparison)
        / comparison.size
    )

    return accuracy

fit(X_train, y_train)

acc = score(
    "merge",
    X_test,
    y_test,
    3
)

print("Accuracy :", round(acc * 100, 2), "%")
print("\n")

#Chatgpt
print("A8\n")

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# ==========================
# COMPARE CUSTOM vs SKLEARN
# ==========================

k_values = list(range(1, 16))

custom_acc = []
sklearn_acc = []

for k in k_values:

    # ---------- Custom KNN ----------
    c_acc = score(
        "merge",
        X_test,
        y_test,
        k
    )

    custom_acc.append(c_acc * 100)

    # ---------- Sklearn KNN ----------
    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    s_acc = model.score(
        X_test,
        y_test
    )

    sklearn_acc.append(s_acc * 100)

# ==========================
# RESULT TABLE
# ==========================

print(f"{'k':<5}{'Custom':<15}{'Sklearn':<15}")

for k, c, s in zip(
    k_values,
    custom_acc,
    sklearn_acc
):
    print(
        f"{k:<5}{c:<15.2f}{s:<15.2f}"
    )

# ==========================
# PLOT
# ==========================

plt.figure(figsize=(10,6))

plt.plot(
    k_values,
    custom_acc,
    marker='o',
    linewidth=2,
    label="Custom KNN"
)

plt.plot(
    k_values,
    sklearn_acc,
    marker='s',
    linewidth=2,
    label="Sklearn KNN"
)

plt.xlabel("k Value")
plt.ylabel("Accuracy (%)")
plt.title("Custom kNN vs Sklearn KNeighborsClassifier")
plt.xticks(k_values)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.show()
print("\n")

#Chatgpt
print("A9\n")
# ==========================
# WEIGHTED KNN ACCURACY
# ==========================

def W_score(value, X_test, y_test, k):

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    predictions = []

    for row in X_test:

        predictions.append(
            w_predict(
                value,
                row,
                k
            )
        )

    correct = np.sum(
        np.array(predictions) == y_test
    )

    return correct / len(y_test)

#Compare Normal kNN and Weighted kNN

k_values = list(range(1, 16))

knn_acc = []
weighted_acc = []

for k in k_values:

    # Normal KNN
    acc1 = score(
        "merge",
        X_test,
        y_test,
        k
    )

    knn_acc.append(acc1 * 100)

    # Weighted KNN
    acc2 = W_score(
        "merge",
        X_test,
        y_test,
        k
    )

    weighted_acc.append(acc2 * 100)

#Display Accuracy Table
print(f"{'k':<5}{'Normal KNN':<15}{'Weighted KNN':<15}")

for k, a1, a2 in zip(
    k_values,
    knn_acc,
    weighted_acc
):
    print(
        f"{k:<5}{a1:<15.2f}{a2:<15.2f}"
    )

#Plot Comparison
plt.figure(figsize=(10,6))

plt.plot(
    k_values,
    knn_acc,
    marker='o',
    linewidth=2,
    label='Normal KNN'
)

plt.plot(
    k_values,
    weighted_acc,
    marker='s',
    linewidth=2,
    label='Weighted KNN'
)

plt.xlabel("Value of k")
plt.ylabel("Accuracy (%)")
plt.title("Normal KNN vs Weighted KNN")

plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)

plt.legend()

plt.show()

#Best k for Each Method
best_knn_k = k_values[
    knn_acc.index(max(knn_acc))
]

best_weighted_k = k_values[
    weighted_acc.index(max(weighted_acc))
]

print(
    "\nBest Normal KNN :",
    best_knn_k,
    "Accuracy =",
    round(max(knn_acc), 2),
    "%"
)

print(
    "Best Weighted KNN :",
    best_weighted_k,
    "Accuracy =",
    round(max(weighted_acc), 2),
    "%"
)
print("\n")

print("Test cases\n")

class TestKNNFunctions(unittest.TestCase):

    # ==========================
    # PREPROCESSING
    # ==========================

    def test_fill_missing(self):

        df = pd.DataFrame({
            "A": [1, np.nan, 3],
            "B": ["x", None, "x"]
        })

        result = fill_missing(df)

        self.assertFalse(result.isnull().values.any())

    def test_label_encoding(self):

        col = pd.Series(["A", "B", "A", "C"])

        encoded, mapping = label_encoding(col)

        self.assertEqual(len(mapping), 3)
        self.assertEqual(encoded[0], encoded[2])

    def test_one_hot_encoding(self):

        col = pd.Series(
            ["Red", "Blue", "Red"],
            name="Color"
        )

        encoded = one_hot_encoding(col)

        self.assertEqual(encoded.shape[1], 2)

    # ==========================
    # DISTANCE
    # ==========================

    def test_eu_distance(self):

        a = [0, 0]
        b = [3, 4]

        self.assertEqual(
            eu_distance(a, b),
            5.0
        )

    # ==========================
    # SORTING
    # ==========================

    def test_bubble_sort(self):

        arr = [(5,1),(2,0),(8,1)]

        result = bubble_sort(arr)

        self.assertEqual(
            result,
            [(2,0),(5,1),(8,1)]
        )

    def test_insertion_sort(self):

        arr = [(4,1),(1,0),(6,1)]

        result = insertion_sort(arr)

        self.assertEqual(
            result,
            [(1,0),(4,1),(6,1)]
        )

    def test_merge_sort(self):

        arr = [(7,0),(3,1),(5,0)]

        result = merge_sort(arr)

        self.assertEqual(
            result,
            [(3,1),(5,0),(7,0)]
        )

    # ==========================
    # VOTING
    # ==========================

    def test_most_number_class(self):

        classes = [1,1,0,1,0]

        result = most_number_class(classes)

        self.assertEqual(result, 1)

    def test_weighted_class(self):

        votes = [
            (0,2.0),
            (1,8.0),
            (1,3.0)
        ]

        result = weighted_class(votes)

        self.assertEqual(result, 1)

    # ==========================
    # MODEL TRAINING
    # ==========================

    def test_fit(self):

        X = np.array([
            [1,2],
            [3,4]
        ])

        y = np.array([0,1])

        result = fit(X,y)

        self.assertEqual(
            result["samples"],
            2
        )

    # ==========================
    # PREDICTION
    # ==========================

    def test_predict(self):

        X = np.array([
            [1,1],
            [2,2],
            [9,9]
        ])

        y = np.array([0,0,1])

        fit(X,y)

        pred = predict(
            "merge",
            [1.5,1.5],
            3
        )

        self.assertEqual(pred, 0)

    def test_weighted_predict(self):

        X = np.array([
            [1,1],
            [2,2],
            [9,9]
        ])

        y = np.array([0,0,1])

        fit(X,y)

        pred = w_predict(
            "merge",
            [1.2,1.2],
            3
        )

        self.assertEqual(pred, 0)

    # ==========================
    # ACCURACY
    # ==========================

    def test_score(self):

        X = np.array([
            [1,1],
            [2,2],
            [9,9],
            [10,10]
        ])

        y = np.array([0,0,1,1])

        fit(X,y)

        acc = score(
            "merge",
            X,
            y,
            1
        )

        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_weighted_score(self):

        X = np.array([
            [1,1],
            [2,2],
            [9,9],
            [10,10]
        ])

        y = np.array([0,0,1,1])

        fit(X,y)

        acc = W_score(
            "merge",
            X,
            y,
            1
        )

        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    # ==========================
    # NEIGHBOURS
    # ==========================

    def test_get_neighbours(self):

        X_train = np.array([
            [1,1],
            [2,2],
            [8,8]
        ])

        y_train = np.array([0,0,1])

        neighbours = get_neighbours(
            X_train,
            y_train,
            [1.5,1.5],
            2,
            "merge"
        )

        self.assertEqual(
            len(neighbours),
            2
        )


if __name__ == "__main__":
    unittest.main()
print("\n")

