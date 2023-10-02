import csv
import pickle

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42

digits_data = load_digits().data
digits_target = load_digits().target[:, None]
X_train, X_test, y_train, y_test = train_test_split(
    digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE
)

with open("data.pickle", "rb") as f:
    class_estimator = pickle.load(f)

answer = class_estimator.predict(X_test)
accuracy = accuracy_score(y_test, answer)
print("Accuracy of tree is", accuracy)


with open("results.csv", "w", newline="") as csvfile:
    fieldnames = ["reference", "answer"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for ref, ans in zip(y_test, answer):
        writer.writerow({"reference": ref[0], "answer": ans})
