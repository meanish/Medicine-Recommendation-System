import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


test=pd.read_csv("test_data.csv",error_bad_lines=False)
train=pd.read_csv("training_data.csv",error_bad_lines=False)

test.head()
train.head()
train.info()
# train=train.drop('Unnamed: 133',axis=1)
train.head()

y_train=train.prognosis
x_train=train.drop('prognosis',axis=1)
x_train
y_train

y_test=test.prognosis
x_test=test.drop('prognosis',axis=1)
x_test.head()
y_test.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.validation import StrList


class RandomForestClassifier(BaggingClassifier):
    _estimator_type = "classifier"

    _hyperparameters = {
        **BaggingClassifier._get_param_names(),
        "criterion": ["gini", "entropy"],
        "max_depth": [int, None],
        "min_samples_split": int,
        "min_samples_leaf": int,
        "min_weight_fraction_leaf": float,
        "max_features": ["auto", "sqrt", "log2"],
        "max_leaf_nodes": [int, None],
        "min_impurity_decrease": float,
        "ccp_alpha": float,
    }

    _parameter_ranges = {
        "n_estimators": {"type": int, "min": 1},
        "bootstrap": {"type": bool},
        "oob_score": {"type": bool},
        "n_jobs": {"type": int, "min": -1},
        "random_state": {"type": int},
        "verbose": {"type": int},
        "warm_start": {"type": bool},
        "max_samples": {"type": [int, float, None]},
    }

    _parameter_constraints: dict = {
        **BaggingClassifier._get_param_names(),
        **DecisionTreeClassifier._get_param_names(),
         "class_weight": [
            StrList(["balanced_subsample", "balanced"]),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")

    def __init__(
        self,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):
        super().__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha


test=pd.read_csv("test_data.csv",error_bad_lines=False)
train=pd.read_csv("training_data.csv",error_bad_lines=False)

test.head()
train.head()

clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_rf=clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)


ac = accuracy_score(y_test,clf_rf.predict(x_test)) #based on decison tree
ps=precision_score(y_test,clf_rf.predict(x_test), average='weighted')
rs= recall_score(y_test,clf_rf.predict(x_test), average='weighted')
fs= f1_score(y_test,clf_rf.predict(x_test), average='weighted')
print('Accuracy is: ',ac)
print('Precison is: ',ps)
print('Recall_Score is: ',rs)
print('f1_score is: ',fs)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test,clf_rf.predict(x_test), average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred, average='weighted'))




import numpy as np

class DecisionTree:
    def __init__(self):
        self.tree = None
    
    def fit(self, X, y):
        # TODO: implement decision tree algorithm
        pass
    
    def predict(self, X):
        # TODO: implement prediction using decision tree
        pass

class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = [DecisionTree() for _ in range(n_trees)]
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for i in range(self.n_trees):
            # create a random sample of the training data
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # build a decision tree using the sample
            self.trees[i].fit(X_sample, y_sample)
    
    def predict(self, X):
        # make predictions using all the decision trees
        predictions = np.zeros((X.shape[0], self.n_trees))
        for i in range(self.n_trees):
            predictions[:, i] = self.trees[i].predict(X)
        
        # aggregate predictions using majority vote
        return np.argmax(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions), axis=1)
