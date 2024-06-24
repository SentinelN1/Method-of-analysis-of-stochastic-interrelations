import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr, kendalltau, kruskal
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor


### чтение датасета ###
df = pd.read_csv("масв/dataframe.csv")

###################################################################################################

################################# АНАЛИЗ КОЛИЧЕСТВЕННЫХ ПЕРЕМЕННЫХ ################################

###################################################################################################

### гистограмма для целевой переменной ###
x = np.linspace(df.price.min(), df.price.max())

fig, ax = plt.subplots(1, 2)
fig.set_figheight(3)
fig.set_figwidth(9)

ax[0].hist(df.price, bins=20)
ax[0].set_xlabel('price')
ax[0].set_ylabel('count')
ax[0].grid(True)

ax[1].hist(df.price, density=True, bins=20)
ax[1].set_xlabel('price')
ax[1].set_ylabel('density')
ax[1].plot(x, norm.pdf(x, df.price.mean(), df.price.std()), 
           label=f'Norm({round(df.price.mean(), 2)}, {round(df.price.std(), 2)})')
ax[1].grid(True)
ax[1].legend()

plt.show()


### удаление выбросов по правилу трех сигм ###
m = df["price"].mean()
s = df["price"].std()
df = df[(m - 3 * s < df["price"]) & (df["price"] < m + 3 * s)]


### построение гистограмм для количественных переменных ###

x1 = np.linspace(df.price.min(), df.price.max())
x2 = np.linspace(df.diff_days.min(), df.diff_days.max())
x3 = np.linspace(df.distance.min(), df.distance.max())


fig, ax = plt.subplots(3, 2)
fig.set_figheight(7)
fig.set_figwidth(7)


ax[0][0].hist(df.price, bins=20)
ax[0][0].set_xlabel('price')
ax[0][0].set_ylabel('count')
ax[0][0].grid(True)

ax[0][1].hist(df.price, density=True, bins=20)
ax[0][1].set_xlabel('price')
ax[0][1].set_ylabel('density')
ax[0][1].plot(x1, norm.pdf(x1, df.price.mean(), df.price.std()), 
           label=f'Norm({round(df.price.mean(), 2)}, {round(df.price.std(), 2)})')
ax[0][1].grid(True)
ax[0][1].legend()


ax[1][0].hist(df.diff_days, bins=20)
ax[1][0].set_xlabel('diff_days')
ax[1][0].set_ylabel('count')
ax[1][0].grid(True)

ax[1][1].hist(df.diff_days, density=True, bins=20)
ax[1][1].set_xlabel('diff_days')
ax[1][1].set_ylabel('density')
ax[1][1].plot(x2, norm.pdf(x2, df.diff_days.mean(), df.diff_days.std()), 
           label=f'Norm({round(df.diff_days.mean(), 2)}, {round(df.diff_days.std(), 2)})')
ax[1][1].grid(True)
ax[1][1].legend()


ax[2][0].hist(df.distance, bins=20)
ax[2][0].set_xlabel('distance')
ax[2][0].set_ylabel('count')
ax[2][0].grid(True)

ax[2][1].hist(df.distance, density=True, bins=20)
ax[2][1].set_xlabel('distance')
ax[2][1].set_ylabel('density')
ax[2][1].plot(x3, norm.pdf(x3, df.distance.mean(), df.distance.std()), 
           label=f'Norm({round(df.distance.mean(), 2)}, {round(df.distance.std(), 2)})')
ax[2][1].grid(True)
ax[2][1].legend()

plt.show()


### вывод статистик для количественных переменных ###
print('\n\nАНАЛИЗ КОЛИЧЕСТВЕННЫХ ПЕРЕМЕННЫХ\n')

print('СТАТИСТИКИ ДЛЯ ПЕРЕМЕННОЙ PRICE')
print(f'медиана {df.price.median()}')
print(f'ассиметрия {df.price.skew()}')
print(f'эксцесс {df.price.kurtosis()}')
print(df.price.describe())

print('\n\nСТАТИСТИКИ ДЛЯ ПЕРЕМЕННОЙ DIFF_DAYS')
print(f'медиана {df.diff_days.median()}')
print(f'ассиметрия {df.diff_days.skew()}')
print(f'эксцесс {df.diff_days.kurtosis()}')
print(df.diff_days.describe())

print('\n\nСТАТИСТИКИ ДЛЯ ПЕРЕМЕННОЙ DISTANCE')
print(f'медиана {df.distance.median()}')
print(f'ассиметрия {df.distance.skew()}')
print(f'эксцесс {df.distance.kurtosis()}')
print(df.distance.describe())

###################################################################################################

############################# АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ ####################################

###################################################################################################
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

print(depart_weekday_count)
print(depart_weekday_percentage)
print(depart_weekday_total)

plt.title("Распределение дней недели вылета")
plt.bar(weekdays_ru, depart_weekday_count)
plt.xlabel("День недели вылета")
plt.ylabel("Частота")
plt.xticks(rotation=45, ha="right")
plt.grid()
plt.show()

#%% Количество пересадок
n = df["number_of_changes"].nunique()
changes_count = np.zeros(n, dtype=int)
for i in range(n):
    changes_count[i] = df[df["number_of_changes"] == i]["number_of_changes"].count()

changes_total = np.sum(changes_count)
changes_percentage = changes_count / changes_total

print(changes_count)
print(changes_percentage)
print(changes_total)

plt.title("Распределение количетсва пересадок")
plt.bar(range(n), changes_count)
plt.xlabel("Количетсво пересадок")
plt.ylabel("Частота")
plt.xticks(rotation=0, ha='center')
plt.grid()
plt.show()

# %% Количество пересадок после укрупнения
df["number_of_changes_category"] = df["number_of_changes"].astype(str)
df.loc[df["number_of_changes"] >= 3, "number_of_changes_category"] = "3+"
df.drop(columns=["number_of_changes"], inplace=True)

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
plt.grid()
plt.show()

# %%
#df.to_csv("df.csv")

# %%


###################################################################################################
# ТУТ ПОКА НЕТ СТАТИСТИК, ТОЛЬКО ГРАФИКИ
############################# АНАЛИЗ СВЯЗИ ЦЕЛЕВАЯ - КАЧЕСТВЕННАЯ ################################

###################################################################################################

### отрисовка графиков (цена - день покупки)

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.search_weekday, bins=20, color='purple', alpha=0.4)
ax1.set_xlabel('search_weekday')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('День недели покупки')

ax2.hist(df.price, bins=20, color='purple', alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='search_weekday', y='price', data=df, ax=ax3,  color='purple', showfliers=None)
ax3.set_xlabel('search_weekday')
ax3.set_ylabel('price')
ax3.set_title("Связь цены с днем недели покупки")
ax3.grid(True)

plt.show()


### отрисовка графиков (цена - день вылета)

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.depart_weekday, bins=20, color='blue', alpha=0.4)
ax1.set_xlabel('depart_weekday')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('День недели вылета')

ax2.hist(df.price, bins=20, color='blue', alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='depart_weekday', y='price', data=df, ax=ax3,  color='blue', showfliers=None)
ax3.set_xlabel('depart_weekday')
ax3.set_ylabel('price')
ax3.set_title("Связь цены с днем недели вылета")
ax3.grid(True)

plt.show()

### отрисовка графиков (цена - колво пересадок)

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.number_of_changes_category, bins=20, color='green', alpha=0.4)
ax1.set_xlabel('number_of_changes_category')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Количество пересадок')

ax2.hist(df.price, bins=20, color='green', alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='number_of_changes_category', y='price', data=df, ax=ax3,  color='green', showfliers=None)
ax3.set_xlabel('number_of_changes_category')
ax3.set_ylabel('price')
ax3.set_title("Связь цены с количеством пересадок")
ax3.grid(True)

plt.show()

### отрисовка графиков (цена - пункт назначения)

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
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
ax3.set_xlabel('destination_category')
ax3.set_ylabel('price')
ax3.set_title("Связь цены с пунктом назначения")
ax3.grid(True)

plt.show()

### отрисовка графиков (цена - авиакомпания)

fig = plt.figure(figsize=(6, 8))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.airline_category, bins=20, color='orange', alpha=0.4)
ax1.set_xlabel('airline_category')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax1.set_title('Авиакомпания')

ax2.hist(df.price, bins=20, color='orange', alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

sns.boxplot(x='airline_category', y='price', data=df, ax=ax3,  color='orange', showfliers=None)
ax3.set_xlabel('airline_category')
ax3.set_ylabel('price')
ax3.set_title("Связь цены с авиакомпанией")
ax3.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
ax3.grid(True)

plt.show()

### подсчет статистик

grouped_categorical_columns = [
    'airline_category',
    'destination_category',
    'number_of_changes_category',
    'depart_weekday',
    'search_weekday'
]

for i in range(5):
    col = grouped_categorical_columns[i]
    subsets = []
    for val in df[col].unique():
        subsets.append(df["price"][df[col] == val])
    print(f'Kruskal test for {col}')
    test_res = kruskal(*subsets)
    print(f'p-value = {test_res.pvalue:.2e}')
    print(f'Null hypothesis is rejected: {test_res.pvalue < 0.05}')
    print()

###################################################################################################

############################# АНАЛИЗ СВЯЗИ ЦЕЛЕВАЯ - КОЛИЧЕСТВЕННАЯ ###############################

###################################################################################################

### связь расстояния и цены ###

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.distance, bins=20, alpha=0.4)
ax1.set_xlabel('distance')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Расстояние')

ax2.hist(df.price, bins=20, alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')

ax3.scatter(df['price'], df['distance'], s=5, alpha=0.2)
ax3.set_xlabel('price')
ax3.set_ylabel('distance')
ax3.set_title("Связь цены с расстоянием")

plt.show()

# Подсчет коэффициентов корреляции
pearson_corr_distance, pearson_p_value_distance = pearsonr(df['price'], df['distance'])
spearman_corr_distance, spearman_p_value_distance = spearmanr(df['price'], df['distance'])
kendall_corr_distance, kendall_p_value_distance = kendalltau(df['price'], df['distance'])

# Вывод результатов
print("\n\nКОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ МЕЖДУ ПЕРЕМЕННЫМИ PRICE И DISTANCE:\n")
print(f"Коэффициент корреляции Пирсона: {pearson_corr_distance}, p-value: {pearson_p_value_distance:.5f}")
print(f"Коэффициент корреляции Спирмена: {spearman_corr_distance}, p-value: {spearman_p_value_distance:.5f}")
print(f"Коэффициент корреляции Кендалла тау: {kendall_corr_distance}, p-value: {kendall_p_value_distance:.5f}")


### связь расстояния и количества дней от покупки билета до вылета ###

fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.diff_days, bins=20, color='purple', alpha=0.4)
ax1.set_xlabel('diff_days')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Кол-во дней от покупки до вылета')

ax2.hist(df.price, bins=20, color='purple', alpha=0.4)
ax2.set_xlabel('price')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Цена')


ax3.scatter(df['price'], df['diff_days'], s=5, alpha=0.2, color='purple')
ax3.set_xlabel('price')
ax3.set_ylabel('diff_days')
ax3.set_title("Связь цены с кол-вом дней от покупки до вылета")

plt.show()


# Подсчет коэффициентов корреляции
pearson_corr_days, pearson_p_value_days = pearsonr(df['price'], df['diff_days'])
spearman_corr_days, spearman_p_value_days = spearmanr(df['price'], df['diff_days'])
kendall_corr_days, kendall_p_value_days = kendalltau(df['price'], df['diff_days'])

# Вывод результатов
print("\n\nКОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ МЕЖДУ ПЕРЕМЕННЫМИ PRICE И DIFF_DAYS:\n")
print(f"Коэффициент корреляции Пирсона: {pearson_corr_days}, p-value: {pearson_p_value_days:.2e}")
print(f"Коэффициент корреляции Спирмена: {spearman_corr_days}, p-value: {spearman_p_value_days:.2e}")
print(f"Коэффициент корреляции Кендалла тау: {kendall_corr_days}, p-value: {kendall_p_value_days:.2e}")
# %%
# df.to_csv("df.csv")

###################################################################################################

############################ АНАЛИЗ СВЯЗИ КАЧЕСТВЕННАЯ - КАЧЕСТВЕННАЯ #############################

###################################################################################################

grouped_categorical_columns = [
    'airline_category',
    'destination_category',
    'number_of_changes_category',
    'depart_weekday',
    'search_weekday'
]

### подсчет статистик
def cramer_v(confusion_matrix, chi2_stat):
    n_observations = confusion_matrix.sum()
    min_shape = min(*confusion_matrix.shape)
    v_cramer_stat = np.sqrt(chi2_stat / (min_shape * n_observations))
    return v_cramer_stat

for left, right in itertools.combinations(grouped_categorical_columns, 2):
    confusion_matrix = pd.crosstab(df[left], df[right]).to_numpy()
    res = sps.chi2_contingency(confusion_matrix)
    chi2_stat, pval = res.statistic, res.pvalue
    v_cramer_stat = cramer_v(confusion_matrix, chi2_stat)
    print(f'{left} to {right}:')
    print(f'Chi Squared statistic = {chi2_stat:.4f}')
    print(f'Chi Squared test p_value = {pval:.4f}')
    print(f'Null hypothesis rejected: {pval < 0.05}')
    print(f'Cramer\'s V = {v_cramer_stat:.4f}')
    print(pd.crosstab(df[left], df[right], margins=True))
    print('=' * 60)

### отрисовка графиков
fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(10, 18))
for i, (left, right) in enumerate(itertools.combinations(grouped_categorical_columns, 2)):
    sns.histplot(data=df, x=left, hue=right, ax=axes[i%5][i//5], multiple="stack")
axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=30, ha='right')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=30, ha='right')
axes[2, 0].set_xticklabels(axes[2, 0].get_xticklabels(), rotation=30, ha='right')
axes[3, 0].set_xticklabels(axes[3, 0].get_xticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.show()

###################################################################################################

############################# АНАЛИЗ СВЯЗИ КАЧЕСТВЕННАЯ - ЧИСЛОВАЯ ###############################

###################################################################################################

numerical_columns = ['distance', 'diff_days']

### подсчет статистик

cat2num_dicts = []
for col in grouped_categorical_columns:
    for num_col in numerical_columns:
        subsets = []
        for val in df[col].unique():
            subsets.append(df[num_col][df[col] == val])
        test_res = sps.kruskal(*subsets)
        cat2num_dicts.append({
            'numerical': num_col,
            'categorical': col,
            'p-value': test_res.pvalue,
            'rejected': test_res.pvalue < 0.05,
            'statistic': test_res.statistic,
        })

print(pd.DataFrame(cat2num_dicts).set_index(['categorical', 'numerical']))

### отрисовка графиков

for cat_col in grouped_categorical_columns:
    fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
    sns.boxplot(data=df, x=cat_col, y='distance', hue=cat_col, ax=axes[0], whis=(0, 100))
    sns.boxplot(data=df, x=cat_col, y='diff_days', hue=cat_col, ax=axes[1], whis=(0, 100))
    if cat_col=='airline_category':
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
        axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha='right')
        
plt.show()


###################################################################################################

################################# АНАЛИЗ СВЯЗИ ЧИСЛОВАЯ - ЧИСЛОВАЯ ################################

###################################################################################################

### построение графика
fig = plt.figure(figsize=(6, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax1.hist(df.distance, bins=20, alpha=0.4)
ax1.set_xlabel('distance')
ax1.set_ylabel('count')
ax1.grid(True)
ax1.set_title('Расстояние')

ax2.hist(df.price, bins=20, alpha=0.4)
ax2.set_xlabel('diff_days')
ax2.set_ylabel('count')
ax2.grid(True)
ax2.set_title('Кол-во дней от покупки до вылета')

ax3.scatter(df['diff_days'], df['distance'], s=5, alpha=0.2)
ax3.set_xlabel('diff_days')
ax3.set_ylabel('distance')
ax3.set_title("Связь кол-ва дней от покупки до вылета с расстоянием")

plt.show()

# Подсчет коэффициентов корреляции
pearson_corr, pearson_p_value = pearsonr(df['diff_days'], df['distance'])
spearman_corr, spearman_p_value = spearmanr(df['diff_days'], df['distance'])
kendall_corr, kendall_p_value = kendalltau(df['diff_days'], df['distance'])

print("\n\nКОЭФФИЦИЕНТЫ КОРРЕЛЯЦИИ МЕЖДУ ПЕРЕМЕННЫМИ DIFF_DAYS И DISTANCE:\n")
print(f"Коэффициент корреляции Пирсона: {pearson_corr}, p-value: {pearson_p_value:.5f}")
print(f"Коэффициент корреляции Спирмена: {spearman_corr}, p-value: {spearman_p_value:.5f}")
print(f"Коэффициент корреляции Кендалла тау: {kendall_corr}, p-value: {kendall_p_value:.5f}")


###################################################################################################

########################################### МОДЕЛИРОВАНИЕ #########################################

###################################################################################################


####### базовая модель

df = df[['search_weekday', 'depart_weekday', 'diff_days', 'distance', 'price', 'number_of_changes_category', 'airline_category', 'destination_category']]

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

# # Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели на тренировочных данных
model.fit(X_train_scaled, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test_scaled)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nБАЗОВАЯ МОДЕЛЬ\n')
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

coeffs = model.coef_
print(f'a_18 = {coeffs[17]}')
print('Коэффициенты модели:')
print(coeffs)


############## анализ наличия гетероскедастичности
# Добавляем константу к матрице предикторов
X_train_const = sm.add_constant(X_train_scaled)

# Создаем модель линейной регрессии
model = sm.OLS(y_train, X_train_const)
results = model.fit()

# Выполняем тест Уайта
white_test = het_white(results.resid, X_train_const)
print(f"LM Statistic: {white_test[0]}")
print(f"P-value: {white_test[1]:.1e}")
print(f"F-statistic: {white_test[2]}")
print(f"P-value for F-statistic: {white_test[3]:.1e}")


############ модифицированная модель 1 (для гипотезы о внутренних/заграничных)

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
y2 = df[target_columns]

X2 = pd.concat([dummies, numerical_values], axis=1)
X2 = sm.add_constant(X2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, train_size=0.8, random_state=11)

# Нормировка признаков
scaler = StandardScaler()
X_train_scaled2 = scaler.fit_transform(X_train2)
X_test_scaled2 = scaler.transform(X_test2)

# # Создание модели линейной регрессии
model2 = LinearRegression()
# Обучение модели на тренировочных данных
model2.fit(X_train_scaled2, y_train2)
# Предсказание на тестовых данных
y_pred2 = model2.predict(X_test_scaled2)

# Оценка качества модели

mse2 = mean_squared_error(y_test2, y_pred2)
rmse2 = root_mean_squared_error(y_test2, y_pred2)
r22 = r2_score(y_test2, y_pred2)

print('\nМОДЕЛЬ ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ 1 (ПРО ВНУТРЕННИЕ/ЗАГРАНИЧНЫЕ ПЕРЕЛЕТЫ)\n')
print(f"Mean Squared Error (MSE): {mse2}")
print(f"Root Mean Squared Error (RMSE): {rmse2}")
print(f"R-squared (R2): {r22}")

coeffs2 = model2.coef_[0]
print(f'a_18_0 (dist_domestic) = {coeffs2[18]}, a_18_1 (dist_abroad) = {coeffs2[19]}')
print('Коэффициенты модели:')
print(coeffs2)


#################### модифицированная модель 2 (анализ гипотезы про дни)

df["diff_days_squared"] = df["diff_days"]**2

categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y3 = df[target_columns]

X3 = pd.concat([dummies, numerical_values], axis=1)
X3 = sm.add_constant(X3)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, train_size=0.8, random_state=11)

# Нормировка признаков
scaler = StandardScaler()
X_train_scaled3 = scaler.fit_transform(X_train3)
X_test_scaled3 = scaler.transform(X_test3)

# # Создание модели линейной регрессии
model3 = LinearRegression()
# Обучение модели на тренировочных данных
model3.fit(X_train_scaled3, y_train3)
# Предсказание на тестовых данных
y_pred3 = model3.predict(X_test_scaled3)

# Оценка качества модели

mse3 = mean_squared_error(y_test3, y_pred3)
rmse3 = root_mean_squared_error(y_test3, y_pred3)
r23 = r2_score(y_test3, y_pred3)

print('\nМОДЕЛЬ ДЛЯ СЛОЖНОЙ ГИПОТЕЗЫ 2 (ПРО КОЛ_ВО ДНЕЙ)\n')
print(f"Mean Squared Error (MSE): {mse3}")
print(f"Root Mean Squared Error (RMSE): {rmse3}")
print(f"R-squared (R2): {r23}")

coeffs3 = model3.coef_[0]
print(f'a_18_0 (diff_days) = {coeffs3[17]}, a_18_1 (diff_days_squared) = {coeffs3[18]}')
print('Коэффициенты модели:')
print(coeffs3)

##### итоговая модель

categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance_domestic", "distance_abroad"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y4 = df[target_columns]

X4 = pd.concat([dummies, numerical_values], axis=1)
X4 = sm.add_constant(X4)

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, train_size=0.8, random_state=11)

# Нормировка признаков
scaler = StandardScaler()
X_train_scaled4 = scaler.fit_transform(X_train4)
X_test_scaled4 = scaler.transform(X_test4)

# # Создание модели линейной регрессии
model4 = LinearRegression()
# Обучение модели на тренировочных данных
model4.fit(X_train_scaled4, y_train4)
# Предсказание на тестовых данных
y_pred4 = model4.predict(X_test_scaled4)

# Оценка качества модели

mse4 = mean_squared_error(y_test4, y_pred4)
rmse4 = root_mean_squared_error(y_test4, y_pred4)
r24 = r2_score(y_test4, y_pred4)


print('\nИТОГОВАЯ МОДЕЛЬ\n')
print(f"Mean Squared Error (MSE): {mse4}")
print(f"Root Mean Squared Error (RMSE): {rmse4}")
print(f"R-squared (R2): {r24}")

coeffs4 = model4.coef_[0]
print(f'a_18_0 (diff_days) = {coeffs4[17]}, a_18_1 (diff_days_squared) = {coeffs4[18]}')
print('Коэффициенты модели:')
print(coeffs4)


################# модифицированная модель

categorical_columns = ["search_weekday", "depart_weekday", "number_of_changes_category", "airline_category", "destination_category"]
numerical_columns = ["diff_days", "diff_days_squared", "distance_domestic", "distance_abroad"]
target_columns = ["price"]

dummies = pd.get_dummies(data=df[categorical_columns], drop_first=True, dtype=int)
numerical_values = df[numerical_columns]
y5 = df[target_columns]

X5 = pd.concat([dummies, numerical_values], axis=1)
X5.drop(columns=["airline_category_Uzbekistan Airways"], inplace=True)

X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, train_size=0.8, random_state=11)

X_train_scaled5 = pd.DataFrame(scaler.fit_transform(X_train5), columns=X_train5.columns, index=X_train5.index)
X_test_scaled5 = pd.DataFrame(scaler.fit_transform(X_test5), columns=X_test5.columns, index=X_test5.index)

X_train5 = sm.add_constant(X_train5)
X_test5 = sm.add_constant(X_test5)
X_train_scaled5 = sm.add_constant(X_train_scaled5)
X_test_scaled5 = sm.add_constant(X_test_scaled5)

model5 = sm.OLS(y_train5, X_train_scaled5)
res5 = model5.fit()
res5.summary()
