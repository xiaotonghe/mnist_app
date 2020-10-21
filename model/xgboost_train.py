from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import xgboost as xgb


train_df = pd.read_csv(
    "/Users/xiaotonghe/Downloads/mnist-flask/data/train.csv")
test_df = pd.read_csv("/Users/xiaotonghe/Downloads/mnist-flask/data/test.csv")


sc = StandardScaler()
X_std = sc.fit_transform(train_df.values[:, 1:])
y = train_df.values[:, 0]

test_std = sc.fit_transform(test_df.values)

X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.1)
param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective",
                                                                                               "multi:softmax"), ("eval_metric", "merror"), ("alpha", 8), ("lambda", 2), ("num_class", 10)]
n_rounds = 600
early_stopping = 50

d_train = xgb.DMatrix(X_train, label=y_train)
d_val = xgb.DMatrix(X_valid, label=y_valid)
eval_list = [(d_train, "train"), (d_val, "validation")]
bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list,
                early_stopping_rounds=early_stopping, verbose_eval=True)
