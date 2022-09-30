import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

k = 2  # pca feature number


def loadData(X_trainPath, X_testPath):
    all_data = pd.read_csv(X_trainPath)
    x_test = pd.read_csv(X_testPath)
    y_train = all_data["class"]
    x_train = pd.read_csv(X_trainPath)
    x_train.drop("class", axis=1, inplace=True)
    return x_train, y_train, x_test


def preprocessing(X, Y, X_Test):
    selector_threshold = VarianceThreshold(0.001)
    selector_threshold.fit(X)

    X_new = selector_threshold.transform(X)
    X_Test_new = selector_threshold.transform(X_Test)

    selector = SelectKBest(chi2, k=10)
    selector.fit(X_new, Y)

    X_new = selector.transform(X_new)
    X_Test_new = selector.transform(X_Test_new)

    pca = PCA(n_components=k, whiten=True)
    pca.fit(X_new)

    X_new = pca.transform(X_new)
    X_Test_new = pca.transform(X_Test_new)
    X_new = Scaler(X_new)

    print("variance_ratio:", pca.explained_variance_ratio_)
    print(("sum of variance_ratio:", sum(pca.explained_variance_ratio_)))
    return X_new, X_Test_new


def Scaler(X):
    sc = StandardScaler()
    X_new = sc.fit_transform(X)
    return X_new


def train_model(X, Y):
    i = 0
    best_models = []
    names = []
    for clf in classifiers:
        model = clf
        model_gscv = GridSearchCV(model, param_grids[i], cv=5)
        model_gscv.fit(x_train_new, y_train)
        best_models.append(model_gscv.best_estimator_)
        name = clf.__class__.__name__
        names.append(name)
        print("=" * 30)
        print(name)
        print(model_gscv.best_params_)
        print('****Results****')
        cv_scores = cross_val_score(model_gscv, x_train_new, y_train, cv=5)
        print("cv_scores mean/std: {}/{}".format(np.mean(cv_scores), np.std(cv_scores)))
        i += 1
    print("=" * 30)

    estimators = [('knn', best_models[0]), ('SVC', best_models[1]), ('DT', best_models[2]), ('LA', best_models[3]),
                  ('QA', best_models[4])]

    ensemble = VotingClassifier(estimators, voting='hard')

    ensemble.fit(x_train_new, y_train)
    cv_scores = cross_val_score(ensemble, x_train_new, y_train, cv=5)
    print("cv_scores mean/std: {}/{}".format(np.mean(cv_scores), np.std(cv_scores)))
    return ensemble


def Predict(model, X):
    return model.predict(X)


def write_output(prediction, FileName):
    ID = np.arange(1, len(prediction) + 1)
    Id_Predict = list(zip(ID, prediction))
    Id_Predict = pd.DataFrame(Id_Predict, columns=['ID', 'Predicted'])
    Id_Predict.to_csv(FileName, index=False)


DTC = DecisionTreeClassifier(random_state=11, max_features="auto", class_weight="balanced", max_depth=None)
param_grids = [
    {'n_neighbors': np.arange(1, 30, 2),
     },
    {
        'kernel': ['rbf', 'linear'],
        'C': np.arange(0.025, 5, 0.025)},
    {
        'max_depth': np.arange(3, 10)},

    {
        'tol': [1e-4]
    },
    {
        'tol': [1.0e-4]
    }
]
classifiers = [
    KNeighborsClassifier(),
    SVC(probability=True),
    DecisionTreeClassifier(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
#####################################################

x_train, y_train, x_test = loadData("train.csv", "test.csv")
########################################################3

x_train_new, x_test_new = preprocessing(x_train, y_train, x_test)

print(x_train_new.shape)
print(x_test_new.shape)
############################################
model = train_model(x_train_new, y_train)
train_predictions = Predict(model, x_test_new)
write_output(train_predictions, "SubmissionEnsemble.csv")
