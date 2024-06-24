#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

#%%
df = pd.read_csv("dataframe.csv")

#%%
weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekdays_ru = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"] 

#%% День недели покупки
search_weekday_count = np.zeros(7)
for i in range(7):
    search_weekday_count[i] = df[df["search_weekday"] == i]["search_weekday"].count()


search_weekday_total = np.sum(search_weekday_count)
search_weekday_percentage = search_weekday_count / search_weekday_total

print(search_weekday_count)
print(search_weekday_percentage)
print(search_weekday_total)

plt.title("Распределение дней недели покупки билета")
plt.bar(weekdays_ru, search_weekday_count)
plt.xlabel("День недели покупки билета")
plt.ylabel("Частота")
plt.xticks(rotation=45, ha="right")
plt.grid()
plt.show()

#%% День недели вылета
depart_weekday_count = np.zeros(7)
for i in range(7):
    depart_weekday_count[i] = df[df["depart_weekday"] == i]["depart_weekday"].count()

depart_weekday_total = np.sum(depart_weekday_count)
depart_weekday_percentage = depart_weekday_count / depart_weekday_total

print(depart_weekday_total)
print(depart_weekday_count)
print(depart_weekday_percentage)

plt.title("Распределение дней недели вылета")
plt.bar(weekdays_ru, depart_weekday_count)
plt.xlabel("День недели вылета")
plt.ylabel("Частота")
plt.xticks(weekdays_ru, rotation=45, ha="right")
plt.grid()
plt.show()

#%% Количество пересадок
changes = df["number_of_changes"].value_counts()

print(np.sum(changes))
print(changes)
print(changes / np.sum(changes))

plt.title("Распределение количетсва пересадок")
plt.bar(changes.index, changes)
plt.xlabel("Количетсво пересадок")
plt.ylabel("Частота")
plt.xticks(rotation=0, ha='center')
plt.grid()
plt.show()

# %% Количество пересадок после укрупнения
df["number_of_changes_category"] = df["number_of_changes"].astype(str)
df.loc[df["number_of_changes"] >= 3, "number_of_changes_category"] = "3+"
# df.drop(columns=["number_of_changes"], inplace=True)
# df.rename(columns={"number_of_changes_category" : "number_of_changes"}, inplace=True)

category = ["0", "1", "2", "3+"]
changes_count = np.zeros(4)
for i, group in enumerate(category):
    changes_count[i] = df[df["number_of_changes_category"] == group]["number_of_changes_category"].count()

changes_total = np.sum(changes_count)
changes_percentage = changes_count / changes_total

print(changes_count)
print(changes_percentage)
print(changes_total)

plt.title("Распределение количетсва пересадок")
plt.bar(category, changes_count)
plt.xlabel("Количетсво пересадок")
plt.ylabel("Частота")
plt.xticks(rotation=0, ha='center')
plt.grid()
plt.show()

# %% Авиакомпании
airlines = df["airline"].value_counts()

print(np.sum(airlines))
print(airlines)
print(airlines / np.sum(airlines))

plt.title("Распределение авиакомпаний")
plt.bar(airlines.index, airlines)
plt.xlabel("Авиакомпания")
plt.ylabel("Частота")
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.show()

# %% Авиакомпании после укрупнения
df["airline_category"] = df["airline"]
df.loc[df.airline.isin(["Wizz Air", "KLM"]), "airline_category"] = "Other"

airlines = df["airline_category"].value_counts()
print(np.sum(airlines))
print(airlines)
print(airlines / np.sum(airlines))

plt.title("Распределение авиакомпаний")
plt.bar(airlines.index, airlines)
plt.xlabel("Авиакомпания")
plt.ylabel("Частота")
plt.xticks(rotation=45, ha='right')
plt.grid()
plt.show()

# %% Страны
countries = df["destination_country_code"].value_counts()

print(np.sum(countries))
print(countries)
print(countries / np.sum(countries))

plt.title("Распределение стран пунктов назначения рейсов")
plt.bar(countries.index, countries)
plt.xlabel("Страна пункта назначения рейса")
plt.ylabel("Частота")
# plt.xticks(rotation=0, ha='right')
plt.grid()
plt.show()

# %% Страны после укрупнения категорий
# Определяем новые категории
domestic = ["RU"]
tourism = ["TR", "ES", "GR", "IN"]
culture = ["DE", "IT", "US"]
cis = ["UZ", "KZ"]

df["destination_category"] = ""
df.loc[df.destination_country_code.isin(domestic), "destination_category"] = "domestic"
df.loc[df.destination_country_code.isin(tourism), "destination_category"] = "tourism"
df.loc[df.destination_country_code.isin(culture), "destination_category"] = "culture"
df.loc[df.destination_country_code.isin(cis), "destination_category"] = "cis"

countries = df["destination_category"].value_counts()

print(np.sum(countries))
print(countries)
print(countries / np.sum(countries))

plt.title("Распределение категорий пунктов назначения рейсов")
plt.bar(countries.index, countries)
plt.xlabel("Категория пункта назначения рейса")
plt.ylabel("Частота")
# plt.xticks(rotation=0, ha='right')
plt.grid()
plt.show()

# %%
# Создание фигуры
fig = plt.figure(figsize=(12, 10))

# Первый график на верхней строке, первый столбец
ax1 = plt.subplot2grid((2, 2), (0, 0))

# Второй график на верхней строке, второй столбец
ax2 = plt.subplot2grid((2, 2), (0, 1))

# Третий график на нижней строке, который занимает оба столбца
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.destination_category, bins=20, color='yellow', alpha=0.6)
ax1.set_xlabel('destination_category')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Пункт назначения')

ax2.hist(df.price, bins=20, color='yellow', alpha=0.6)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='destination_category', y='price', data=df, ax=ax3,  color='yellow', showfliers=None)
ax3.set_xlabel('price')
ax3.set_ylabel('destination_category')
ax3.set_title("Связь цены с пунктом назначения")
ax3.grid(True)

plt.show()

# %%
fig = plt.figure(figsize=(12, 10))

# Первый график на верхней строке, первый столбец
ax1 = plt.subplot2grid((2, 2), (0, 0))

# Второй график на верхней строке, второй столбец
ax2 = plt.subplot2grid((2, 2), (0, 1))

# Третий график на нижней строке, который занимает оба столбца
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.destination_category, bins=20, color='yellow', alpha=0.6)
ax1.set_xlabel('destination_category')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Пункт назначения')

ax2.hist(df.price, bins=20, color='yellow', alpha=0.6)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='destination_category', y='price', data=df, ax=ax3,  color='yellow', showfliers=None)
ax3.set_xlabel('price')
ax3.set_ylabel('destination_category')
ax3.set_title("Связь цены с пунктом назначения")
ax3.grid(True)

plt.show()

# %%
categorical_variables = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_variables = ["distance", "diff_days"]
target_variable = "price"

# %%
for variable in categorical_variables:    
    targets = []
    for value in df[variable].unique():
        targets.append(df[df[variable] == value][target_variable])
    print(variable, stats.kruskal(*targets))

# %%
for variable in numerical_variables:
    print(variable)
    print(stats.pearsonr(df[variable], df[target_variable]))
    print(stats.spearmanr(df[variable], df[target_variable]))
    print(stats.kendalltau(df[variable], df[target_variable]))
    print("\n")

# %%
df.to_json("df.json")

# %%
