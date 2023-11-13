# assignment 04: Decision Tree consttruction
import numpy as np
from sklearn.base import BaseEstimator


r"""__Your ultimate task for today is to impement the `DecisionTree` class
and use it to solve classification and regression problems.__

__Specifications:__
- The class inherits from `sklearn.BaseEstimator`;
- Constructor is implemented for you. It has the following parameters:
    * `max_depth` - maximum depth of the tree; `np.inf` by default
    * `min_samples_split` - minimal number of samples in the leaf to make a
    split;      `2` by default;
    * `criterion` - criterion to select the best split; in classification one
    of `['gini', 'entropy']`, default `gini`; in regression `variance`;

- `fit` method takes `X` (`numpy.array` of type `float` shaped `(n_objects,
n_features)`) and `y` (`numpy.array` of type float shaped `(n_objects, 1)` in
regression; `numpy.array` of type int shaped `(n_objects, 1)` with class
labels in classification). It works inplace and fits the `DecisionTree` class
instance to the provided data from scratch.

- `predict` method takes `X` (`numpy.array` of type `float` shaped
`(n_objects, n_features)`) and returns the predicted $\hat{y}$ values. In
classification it is a class label for every object (the most frequent in the
leaf; if several classes meet this requirement select the one with the
smallest class index). In regression it is the desired constant (e.g. mean
value for `variance` criterion)

- `predict_proba` method (works only for classification (`gini` or `entropy`
criterion). It takes `X` (`numpy.array` of type `float` shaped `(n_objects,
n_features)`) and returns the `numpy.array` of type `float` shaped
`(n_objects, n_features)` with class probabilities for every object from `X`.
Class $i$ probability equals the ratio of $i$ class objects that got in this
node in the training set.


__Small recap:__

To find the optimal split the following functional is evaluated:

$$G(j, t) = H(Q) - \dfrac{|L|}{|Q|} H(L) - \dfrac{|R|}{|Q|} H(R),$$
where $Q$ is the dataset from the current node, $L$ and $R$ are left and right
subsets defined by the split $x^{(j)} < t$.



1. Classification. Let $p_i$ be the probability of $i$ class in subset $X$
(ratio of the $i$ class objects in the dataset). The criterions are defined as:

    * `gini`: Gini impurity $$H(R) = 1 -\sum_{i = 1}^K p_i^2$$

    * `entropy`: Entropy $$H(R) = -\sum_{i = 1}^K p_i \log(p_i)$$ (One might
    use the natural logarithm).

2. Regression. Let $y_l$ be the target value for the $R$, $\mathbf{y} = (y_1,
\dots, y_N)$ - all targets for the selected dataset $X$.

    * `variance`: $$H(R) = \dfrac{1}{|R|} \sum_{y_j \in R}(y_j - \text{mean}
    (\mathbf{y}))^2$$

    * `mad_median`: $$H(R) = \dfrac{1}{|R|} \sum_{y_j \in R}|y_j -
    \text{median}(\mathbf{y})|$$

**Hints and comments**:

* No need to deal with categorical features, they will not be present.
* Simple greedy recursive procedure is enough. However, you can speed it up
somehow (e.g. using percentiles).
* Please, do not copy implementations available online. You are supposed to
build very simple example of the Decision Tree.

File `tree.py` is waiting for you. Implement all the needed methods in that
file.

### Check yourself
"""


def counter(f):
    def wrapped(*args, **kwargs):
        wrapped.depth += 1
        try:
            return f(*args, **kwargs)
        finally:
            wrapped.depth -= 1

    wrapped.depth = 0
    return wrapped


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for
    numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    # YOUR CODE HERE
    proba = y.sum(axis=0) / y.shape[0]
    return -np.sum(np.multiply(proba, np.log2(proba + EPS)))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    # YOUR CODE HERE
    proba = y.sum(axis=0) / y.shape[0]
    return 1 - np.sum(np.square(proba))


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    return np.var(y)


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    return np.mean(np.abs(y - np.median(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.0
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your
    code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        "gini": (gini, True),  # (criterion, classification flag)
        "entropy": (entropy, True),
        "variance": (variance, False),
        "mad_median": (mad_median, False),
    }

    def __init__(
        self,
        n_classes=None,
        max_depth=np.inf,
        min_samples_split=2,
        criterion_name="gini",
        debug=False,
    ):
        assert (
            criterion_name in self.all_criterions.keys()
        ), "Criterion name must be on of the following: {}".format(
            self.all_criterions.keys()
        )

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using
        provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes)
        in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input
        X_subset and y_subset
            Part of the providev subset where selected feature
            x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input
        X_subset and y_subset
            Part of the providev subset where selected feature
            x^j >= threshold
        """

        # YOUR CODE HERE
        X_subset_feature = X_subset[:, feature_index]
        X_subset_feature_bool = X_subset_feature < threshold
        X_left, X_right = (
            X_subset[X_subset_feature_bool],
            X_subset[~X_subset_feature_bool],
        )
        y_left, y_right = (
            y_subset[X_subset_feature_bool],
            y_subset[~X_subset_feature_bool],
        )

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature
        and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes)
        in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes)
        in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape
        (n_objects_right, n_classes)
        in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        X_subset_feature = X_subset[:, feature_index]
        X_subset_feature_bool = X_subset_feature < threshold
        y_left, y_right = (
            y_subset[X_subset_feature_bool],
            y_subset[~X_subset_feature_bool],
        )

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t.
        selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes)
        in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for
            corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        feature_index, threshold = None, None
        max_gain = None
        for index_of_current_feature in range(X_subset.shape[1]):
            set_of_thresholds = set(X_subset[:, index_of_current_feature])
            if len(set_of_thresholds) > 1:
                set_of_thresholds.discard(
                    min(set_of_thresholds)
                )  # split uses "<" therefore min is discard
            else:
                continue
            for current_threshold in set_of_thresholds:
                y_left, y_right = self.make_split_only_y(
                    index_of_current_feature,
                    current_threshold,
                    X_subset,
                    y_subset,
                )
                if y_left.size != 0 and y_right.size != 0:
                    HQ = self.criterion(y_subset)
                    HL = self.criterion(y_left)
                    HR = self.criterion(y_right)
                    LQ = y_left.size / y_subset.size
                    RQ = y_right.size / y_subset.size
                    gain = HQ - LQ * HL - RQ * HR
                    if max_gain is None or gain > max_gain:
                        max_gain = gain
                        feature_index, threshold = (
                            index_of_current_feature,
                            current_threshold,
                        )
                        if max_gain == HQ:
                            return feature_index, threshold

        assert feature_index is not None
        assert threshold is not None
        return feature_index, threshold

    @counter
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape
        (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes)
        in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for
            corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
        # YOUR CODE HERE
        proba = y_subset.sum(axis=0) / y_subset.shape[0]
        if (
            y_subset.shape[0] <= self.min_samples_split
            or self.make_tree.depth > self.max_depth
            or np.allclose(proba.max(), 1.0)
        ):
            self.depth = max(self.depth, self.make_tree.depth)
            return Node(None, None, proba)
        best_feature, best_threshold = self.choose_best_split(
            X_subset, y_subset
        )
        (X_left, y_left), (X_right, y_right) = self.make_split(
            best_feature, best_threshold, X_subset, y_subset
        )

        new_node = Node(best_feature, best_threshold, proba)
        new_node.left_child = self.make_tree(X_left, y_left)
        new_node.right_child = self.make_tree(X_right, y_right)

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target
            values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), "Wrong y shape"
        self.criterion, self.classification = self.all_criterions[
            self.criterion_name
        ]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch
        using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions
            should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1)
        in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target
            values in regression

        """
        # YOUR CODE HERE
        if self.classification:
            probs = self.predict_proba(X)
            y_predicted = np.argmax(probs, axis=1)
        else:
            y_predicted = []  # np.zeros((X.shape[0], 1))
            for sample_index in range(X.shape[0]):
                solver = self.root
                while (
                    solver.feature_index is not None
                ):  # this is a solver, not a leaf
                    feature = solver.feature_index
                    threshold = solver.value
                    if X[sample_index, feature] < threshold:
                        solver = solver.left_child
                    else:
                        solver = solver.right_child
                y_predicted.append(solver.proba)
        return np.asarray(y_predicted)

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape
        (n_objects, n_features)
            Feature matrix representing the data the predictions
            should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape
        (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, "Available only for classification problem"

        # YOUR CODE HERE
        y_predicted_probs = []  # np.zeros((X.shape[0], 10), dtype=np.float64)
        for sample_index in range(X.shape[0]):
            solver = self.root
            while (
                solver.feature_index is not None
            ):  # this is a solver, not a leaf
                feature = solver.feature_index
                threshold = solver.value
                if X[sample_index, feature] < threshold:
                    solver = solver.left_child
                else:
                    solver = solver.right_child
            y_predicted_probs.append(solver.proba)

        return np.asarray(y_predicted_probs)
