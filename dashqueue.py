import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white

from scipy import stats

df=pd.read_json("df.json")

df = df[['search_weekday', 'depart_weekday', 'diff_days', 'distance', 'price', 'number_of_changes_category', 'airline_category', 'destination_category']]

grouped_categorical_columns = ['search_weekday', 'depart_weekday', 'number_of_changes_category', 'airline_category', 'destination_category']
numerical_columns = ['diff_days', 'distance']
y = df['price']

numerical_values = df[numerical_columns]
dummies = pd.get_dummies(df[grouped_categorical_columns], drop_first=True)

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормировка признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание модели линейной регрессии
model = linear_model.LinearRegression()

# Обучение модели на тренировочных данных
model.fit(X_train_scaled, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка качества модели

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")


