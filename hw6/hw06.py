import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

train_X = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw06/training_data.csv")
test_X = pd.read_csv("D:/Downloads/engr421_dasc521_fall2019_hw06/test_data.csv")

train_Y = train_X['TRX_COUNT']

train_X.drop(['TRX_COUNT'], axis=1, inplace=True)

x = pd.concat([train_X, pd.get_dummies(train_X['IDENTITY'], prefix='IDENTITY')], axis=1)
x.drop(['IDENTITY'], axis=1, inplace=True)

x.insert(0, 'WEEKEND',
         x.apply(lambda row: 0 if (pd.Timestamp(row['YEAR'], row['MONTH'], row['DAY']).weekday() <= 5) else 1, axis=1))

x_train = np.array(x)
y_train = np.array(train_Y)

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

model = RandomForestRegressor(n_estimators=50, random_state=0)

model.fit(X_train, Y_train)
y_predict = model.predict(X_test)
rmse = sqrt(mean_squared_error(Y_test, y_predict))
mae = mean_absolute_error(Y_test, y_predict)

print("Cross validation RMSE: ", rmse)
print("Cross validation MAE: ", mae)

y = pd.concat([test_X, pd.get_dummies(test_X['IDENTITY'], prefix='IDENTITY')], axis=1)
y.drop(['IDENTITY'], axis=1, inplace=True)

y.insert(0, 'WEEKEND',
         y.apply(lambda row: 0 if (pd.Timestamp(row['YEAR'], row['MONTH'], row['DAY']).weekday() <= 5) else 1, axis=1))

x_test = np.array(y)

y_predicted = model.predict(x_test)

y_predicted = pd.DataFrame(y_predicted)
y_predicted.to_csv('test_predictions.csv', sep='\t', encoding='utf-8', index=False)
