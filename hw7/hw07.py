import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

X_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target1_training_data.csv")
Y_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target1_training_label.csv")
X_test = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target1_test_data.csv")
X_test_1 = X_test

X_test_id = X_test_1["ID"]

X_train["TARGET"] = Y_train.TARGET
duplicate = X_train[X_train["TARGET"] == 1]
X_train = X_train.append([duplicate]*5, ignore_index=True)

Y_train_duplicated = X_train["TARGET"]

X_train.drop(['TARGET'], axis=1, inplace=True)
X_train.drop(['ID'], axis=1, inplace=True)
X_train = X_train.replace(np.nan, X_train.mean())
X_test.drop(['ID'], axis=1, inplace=True)
X_test = X_test.replace(np.nan, X_test.mean())

cols = X_train.columns
num_cols = X_train._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_train = pd.get_dummies(X_train, categorical_data, dummy_na=True)

cols = X_test.columns
num_cols = X_test._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_test = pd.get_dummies(X_test, categorical_data, dummy_na=True)

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0

missing_cols_test = set(X_test.columns) - set(X_train.columns)
for c in missing_cols_test:
    X_train[c] = 0

X_train = np.array(X_train)
Y_train = np.array(Y_train_duplicated)
X_test_original = np.array(X_test)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0)

model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
auroc = roc_auc_score(Y_predict, Y_test)

print(auroc)

Y_predicted = model.predict(X_test_original)

y_pred_test_df = pd.DataFrame({"ID": X_test_id, "TARGET": Y_predicted})
y_pred_test_df.to_csv("hw07_target1_test_predictions.csv", index=False)

X_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target2_training_data.csv")
Y_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target2_training_label.csv")
X_test = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target2_test_data.csv")
X_test_1 = X_test

X_test_id = X_test_1["ID"]

X_train["TARGET"] = Y_train.TARGET
duplicate = X_train[X_train["TARGET"] == 1]
X_train = X_train.append([duplicate]*16, ignore_index=True)

Y_train_duplicated = X_train["TARGET"]

X_train.drop(['TARGET'], axis=1, inplace=True)
X_train.drop(['ID'], axis=1, inplace=True)
X_train = X_train.replace(np.nan, X_train.mean())
X_test.drop(['ID'], axis=1, inplace=True)
X_test = X_test.replace(np.nan, X_test.mean())

cols = X_train.columns
num_cols = X_train._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_train = pd.get_dummies(X_train, categorical_data, dummy_na=True)

cols = X_test.columns
num_cols = X_test._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_test = pd.get_dummies(X_test, categorical_data, dummy_na=True)

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0

missing_cols_test = set(X_test.columns) - set(X_train.columns)
for c in missing_cols_test:
    X_train[c] = 0

X_train = np.array(X_train)
Y_train = np.array(Y_train_duplicated)
X_test_original = np.array(X_test)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0)

model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
auroc = roc_auc_score(Y_predict, Y_test)

print(auroc)

Y_predicted = model.predict(X_test_original)

y_pred_test_df = pd.DataFrame({"ID": X_test_id, "TARGET": Y_predicted})
y_pred_test_df.to_csv("hw07_target2_test_predictions.csv", index=False)

X_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target3_training_data.csv")
Y_train = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target3_training_label.csv")
X_test = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw07/hw07_target3_test_data.csv")
X_test_1 = X_test

X_test_id = X_test_1["ID"]

X_train["TARGET"] = Y_train.TARGET
duplicate = X_train[X_train["TARGET"] == 1]
X_train = X_train.append([duplicate]*5, ignore_index=True)

Y_train_duplicated = X_train["TARGET"]

X_train.drop(['TARGET'], axis=1, inplace=True)
X_train.drop(['ID'], axis=1, inplace=True)
X_train = X_train.replace(np.nan, X_train.mean())
X_test.drop(['ID'], axis=1, inplace=True)
X_test = X_test.replace(np.nan, X_test.mean())

cols = X_train.columns
num_cols = X_train._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_train = pd.get_dummies(X_train, categorical_data, dummy_na=True)

cols = X_test.columns
num_cols = X_test._get_numeric_data().columns
categorical_data = list(set(cols)-set(num_cols))
X_test = pd.get_dummies(X_test, categorical_data, dummy_na=True)

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0

missing_cols_test = set(X_test.columns) - set(X_train.columns)
for c in missing_cols_test:
    X_train[c] = 0

X_train = np.array(X_train)
Y_train = np.array(Y_train_duplicated)
X_test_original = np.array(X_test)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=50, random_state=0)

model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
auroc = roc_auc_score(Y_predict, Y_test)

print(auroc)

Y_predicted = model.predict(X_test_original)

y_pred_test_df = pd.DataFrame({"ID": X_test_id, "TARGET": Y_predicted})
y_pred_test_df.to_csv("hw07_target3_test_predictions.csv", index=False)