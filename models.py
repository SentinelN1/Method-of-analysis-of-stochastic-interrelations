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


scaler = StandardScaler()


# df = pd.read_json("df.json")
df = pd.read_csv("dataframe.csv")

# %%
categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "distance"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y = df[target_columns]

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled)
res = model.fit()
print(res.summary())

vif_data = pd.DataFrame()
vif_data['feature'] = X_train.columns
vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(vif_data)

white_test = het_white(res.resid, res.model.exog)
labels = ['LM', 'LM p-value', 'F', 'F p-value']
print(dict(zip(labels, white_test)))

y_pred = res.predict(X_test)
print("Среднеквадратическая погрешность:", metrics.root_mean_squared_error(y_test, y_pred))
print("Абсолютная погрешность:", metrics.mean_absolute_error(y_test, y_pred))

# %%
df["diff_days_squared"] = df["diff_days"]**2

categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y = df[target_columns]

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled)
res = model.fit()
print(res.summary())

y_pred = res.predict(X_test)
print("Среднеквадратическая погрешность:", metrics.root_mean_squared_error(y_test, y_pred))
print("Абсолютная погрешность:", metrics.mean_absolute_error(y_test, y_pred))

# %%
df["distance_domestic"] = 0.0
df["distance_abroad"] = 0.0

domestic_flights = df["destination_category"] == "domestic"
df.loc[domestic_flights, "distance_domestic"] = df[domestic_flights]["distance"]

abroad_flights = df["destination_category"] != "domestic"
df.loc[abroad_flights, "distance_abroad"] = df[abroad_flights]["distance"]


categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "distance_domestic", "distance_abroad"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y = df[target_columns]

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled)
res = model.fit()
print(res.summary())

y_pred = res.predict(X_test)
print("Среднеквадратическая погрешность:", metrics.root_mean_squared_error(y_test, y_pred))
print("Абсолютная погрешность:", metrics.mean_absolute_error(y_test, y_pred))

# %%
categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance_domestic", "distance_abroad"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y = df[target_columns]

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled)
res = model.fit()
print(res.summary())

y_pred = res.predict(X_test)
print("Среднеквадратическая погрешность:", metrics.root_mean_squared_error(y_test, y_pred))
print("Абсолютная погрешность:", metrics.mean_absolute_error(y_test, y_pred))


# %%
categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance_domestic", "distance_abroad"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y = df[target_columns]

X = pd.concat([dummies, numerical_values], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=11)

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

model = sm.OLS(y_train, X_train_scaled)
res = model.fit()
print(res.summary())

y_pred = res.predict(X_test)
print("Среднеквадратическая погрешность:", metrics.root_mean_squared_error(y_test, y_pred))
print("Абсолютная погрешность:", metrics.mean_absolute_error(y_test, y_pred))
