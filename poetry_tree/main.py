import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from tree import DecisionTree


"""#### Simple check"""

RANDOM_STATE = 42

X = np.ones((4, 5), dtype=float) * np.arange(4)[:, None]
y = np.arange(4)[:, None] + np.asarray([0.2, -0.3, 0.1, 0.4])[:, None]
class_estimator = DecisionTree(max_depth=10, criterion_name="gini")

(X_l, y_l), (X_r, y_r) = class_estimator.make_split(1, 1.0, X, y)

assert np.array_equal(X[:1], X_l)
assert np.array_equal(X[1:], X_r)
assert np.array_equal(y[:1], y_l)
assert np.array_equal(y[1:], y_r)

"""#### Classification problem"""

digits_data = load_digits().data
digits_target = load_digits().target[
    :, None
]  # to make the targets consistent with our model interfaces
X_train, X_test, y_train, y_test = train_test_split(
    digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE
)

assert len(y_train.shape) == 2 and y_train.shape[0] == len(X_train)

class_estimator = DecisionTree(max_depth=10, criterion_name="gini")
class_estimator.fit(X_train, y_train)

with open("data.pickle", "wb") as f:
    pickle.dump(class_estimator, f)

ans = class_estimator.predict(X_test)
accuracy_gini = accuracy_score(y_test, ans)
print(accuracy_gini)

reference = np.array(
    [
        0.09027778,
        0.09236111,
        0.08333333,
        0.09583333,
        0.11944444,
        0.13888889,
        0.09930556,
        0.09444444,
        0.08055556,
        0.10555556,
    ]
)

class_estimator = DecisionTree(max_depth=10, criterion_name="entropy")
class_estimator.fit(X_train, y_train)
ans = class_estimator.predict(X_test)
accuracy_entropy = accuracy_score(y_test, ans)
print(accuracy_entropy)

assert 0.84 < accuracy_gini < 0.9
assert 0.86 < accuracy_entropy < 0.9
assert (
    np.sum(
        np.abs(class_estimator.predict_proba(X_test).mean(axis=0) - reference)
    )
    < 1e-4
)

np.sum(np.abs(class_estimator.predict_proba(X_test).mean(axis=0) - reference))

"""Let's use 5-fold cross validation (`GridSearchCV`)
to find optimal values
for `max_depth` and `criterion` hyperparameters."""

param_grid = {"max_depth": range(3, 11), "criterion_name": ["gini", "entropy"]}
gs = GridSearchCV(
    DecisionTree(), param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-2
)

# Commented out IPython magic to ensure Python compatibility.
# %%time
gs.fit(X_train, y_train)

gs.best_params_

assert gs.best_params_["criterion_name"] == "entropy"
assert 6 < gs.best_params_["max_depth"] < 9

plt.figure(figsize=(10, 8))
plt.title("The dependence of quality on the depth of the tree")
plt.plot(np.arange(3, 11), gs.cv_results_["mean_test_score"][:8], label="Gini")
plt.plot(
    np.arange(3, 11), gs.cv_results_["mean_test_score"][8:], label="Entropy"
)
plt.legend(fontsize=11, loc=1)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.show()


"""#### Regression problem"""

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
regr_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
regr_target = raw_df.values[1::2, 2]

# regr_data = load_boston().data
regr_target = regr_target[
    :, None
]  # to make the targets consistent with our model interfaces
RX_train, RX_test, Ry_train, Ry_test = train_test_split(
    regr_data, regr_target, test_size=0.2, random_state=RANDOM_STATE
)

regressor = DecisionTree(max_depth=10, criterion_name="mad_median")
regressor.fit(RX_train, Ry_train)
predictions_mad = regressor.predict(RX_test)
mse_mad = mean_squared_error(Ry_test, predictions_mad)
print(mse_mad)

regressor = DecisionTree(max_depth=10, criterion_name="variance")
regressor.fit(RX_train, Ry_train)
predictions_mad = regressor.predict(RX_test)
mse_var = mean_squared_error(Ry_test, predictions_mad)
print(mse_var)

assert 9 < mse_mad < 20
assert 8 < mse_var < 12

param_grid_R = {
    "max_depth": range(2, 9),
    "criterion_name": ["variance", "mad_median"],
}

gs_R = GridSearchCV(
    DecisionTree(),
    param_grid=param_grid_R,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-2,
)
gs_R.fit(RX_train, Ry_train)

gs_R.best_params_

assert gs_R.best_params_["criterion_name"] == "mad_median"
assert 3 < gs_R.best_params_["max_depth"] < 7

var_scores = gs_R.cv_results_["mean_test_score"][:7]
mad_scores = gs_R.cv_results_["mean_test_score"][7:]

plt.figure(figsize=(10, 8))
plt.title("The dependence of neg_mse on the depth of the tree")
plt.plot(np.arange(2, 9), var_scores, label="variance")
plt.plot(np.arange(2, 9), mad_scores, label="mad_median")
plt.legend(fontsize=11, loc=1)
plt.xlabel("max_depth")
plt.ylabel("neg_mse")
plt.show()
