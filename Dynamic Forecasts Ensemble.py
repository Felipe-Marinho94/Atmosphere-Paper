#Ensemble de previsões para o conjunto
#de Petrolina-PE
#Autor:Felipe Pinto Marinho
#Data:16/02/2023

#Carregando algumas bibliotecas
import pandas as pd

#Métodos e divisão
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV 
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import random

#Gráficos
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
from math import pi
import plotly.graph_objects as go
pio.renderers.default = 'browser'

#Carregando a biblioteca src
from src.tde import time_delay_embedding

# windowing 
from src.ensembles.windowing import WindowLoss

# arbitrating (a meta-learning strategy)
from src.ensembles.ade import Arbitrating

#Carregando os dados
#Previsão GHI para horizonte de 10 min a frente
Dados = pd.read_csv('Dados_irradiancia.txt', sep=',')
Dados = Dados.dropna(ignore_index=True)
#Dados = Dados[Dados['ktDNI_10'] != 0]

#Obtenção do I_cls para DNI
Dados['I_cls_DNI_10'] = np.abs(Dados['I_cls_10']/np.cos(Dados['zenith']))

#Previsão somente para GHI e ktGHI
Dados = Dados.drop(columns=['DNI_avg', 'ktDNI_avg', "GHI_10",
                            'DNI_60', 'DNI_360', 'GHI_60',
                            'GHI_360', 'ktGHI_60', 'ktGHI_360',
                            'I_cls_60', 'I_cls_360', 'ktGHI_10',
                            'ktDNI_60', 'ktDNI_360'])

#Trabalhando só com os preditores recursivos LBV
Dados = Dados.drop(columns=['year', 'day', 'min', 'zenith',
                            'GHI_avg', 'ktGHI_avg', 'Temp_avg',
                            'I_cls'])

#Trabalhando só com os preditores L
Dados = Dados.drop(columns=['B1', 'B2', 'B3', 'B4',
                            'B5', 'B6', 'V1', 'V2', 
                            'V3', 'V4', 'V5'])

#Carregando os dados de velocidade do vento
Dados = pd.read_csv('Dados_velocidade.txt', sep = ',')
Dados = Dados.dropna(ignore_index=True)


#Removendo algumas colunas para ficar com os preditores LBV
Dados = Dados.drop(columns = ['datetm', 'year', 'day',
                              'min', 'ws_25', 'wd_25',
                              'tp_25', 'ws_50', 'wd_50',
                              'tp_50', 'seno', 'cosseno',
                              'ws_50_20', 'ws_50_30', 'ws_50_60'])

#Removendo mais colunas para ficar somente com os preditores L
Dados = Dados.drop(columns = ['B1', 'B2', 'B3',
                              'B4', 'B5', 'B6',
                              'V1', 'V2', 'V3',
                              'V4', 'V5', 'V6'])

#Divisão treino-teste
random.seed(10)
train, test = train_test_split(Dados, test_size=0.3, shuffle=False) 
random.seed(10)
train_idx, test_idx = train_test_split(Dados.index, test_size=0.3, shuffle=False)


#Para o conjunto de dados de velocidade do vento
X_train, y_train = train.drop(columns=['ws_50_10']), train[train.columns[6]]
X_test, y_test = test.drop(columns=['ws_50_10']), test[test.columns[6]]

# Pevisão sobre o ktDNI
X_train, y_train = train.drop(columns=['DNI_10', 'ktDNI_10', 'I_cls_10']), train[train.columns[6]]
X_test, y_test = test.drop(columns=['DNI_10', 'ktDNI_10', 'I_cls_10']), test[test.columns[6]]

##############################################################
#Treinando os modelos utilizando validação cruzada
#5 Folds no conjunto de treino
scoring_regressor = {'NRMSE': 'neg_root_mean_squared_error',
                     'NMAE': 'neg_mean_absolute_error'}

#Random Forest
param_grid_RF = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [2, 5, 7, 9, 11, 13, 15, 21, 35]
}

gs_RF = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_RF, scoring=scoring_regressor,
                     cv = 5, refit = 'NRMSE', return_train_score = True)

#X, y = pd.concat([X_train, X_test], axis=0).reset_index(drop=True), pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
gs_RF.fit(X_train, y_train)
results_RF = gs_RF.cv_results_

#Plotando os resultados
plt.rcParams.update({'font.size': 19})
plt.figure(figsize=(13, 13))
plt.xlabel("max_depth")
plt.ylabel("Score")
ax = plt.gca()
ax.set_xlim(0, 25)
ax.set_ylim(0, -10)

X_axis = np.array(results_RF["param_max_depth"].data, dtype=float)

for scorer, color in zip(sorted(scoring_regressor), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results_RF["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results_RF["std_%s_%s" % (sample, scorer)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    best_index = np.nonzero(results_RF["rank_test_%s" % scorer] == 1)[0][0]
    best_score = results_RF["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

###################################################################
#KNN
#Grid para valores de vizinhos
param_grid_KNN = {'n_neighbors': np.arange(1, 50)}
gs_KNN = GridSearchCV(KNeighborsRegressor(), param_grid_KNN, scoring=scoring_regressor,
                     cv = 5, refit = 'NRMSE', return_train_score = True)

gs_KNN.fit(X_train, y_train)
results_KNN = gs_KNN.cv_results_

#Plotando gráficos
plt.figure(figsize=(13, 13))
plt.xlabel("Numbers Neighbors")
plt.ylabel("Score")
ax = plt.gca()
ax.set_xlim(0, 50)
ax.set_ylim(0, -10)

X_axis = np.array(results_KNN["param_n_neighbors"].data, dtype=float)

for scorer, color in zip(sorted(scoring_regressor), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results_KNN["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results_KNN["std_%s_%s" % (sample, scorer)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    best_index = np.nonzero(results_KNN["rank_test_%s" % scorer] == 1)[0][0]
    best_score = results_KNN["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

###########################################################################
#SVR
#Grid para SVR
param_grid_SVR = {'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']}

gs_SVR = GridSearchCV(SVR(), param_grid_SVR, scoring=scoring_regressor,
                     cv = 5, refit = 'NRMSE', return_train_score = True)

gs_SVR.fit(X_train, y_train)
results_SVR = gs_SVR.cv_results_

#Plotando gráficos
plt.figure(figsize=(13, 13))
plt.xlabel("epsilon")
plt.ylabel("Score")
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, -500)

X_axis = np.array(results_SVR["param_epsilon"].data, dtype=float)

for scorer, color in zip(sorted(scoring_regressor), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results_SVR["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results_SVR["std_%s_%s" % (sample, scorer)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    best_index = np.nonzero(results_SVR["rank_test_%s" % scorer] == 1)[0][0]
    best_score = results_SVR["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

#########################################################################
#Elastic Net
param_grid_Elastic = {'l1_ratio': [1, 0.1, 0.01, 0.001, 0.0001]}

gs_Elastic = GridSearchCV(ElasticNetCV(), param_grid_Elastic, scoring=scoring_regressor,
                     cv = 5, refit = 'NRMSE', return_train_score = True)

gs_Elastic.fit(X_train, y_train)
results_Elastic = gs_Elastic.cv_results_

#Plotando figuras
plt.figure(figsize=(13, 13))
plt.xlabel("Lambda")
plt.ylabel("Score")
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, -500)

for scorer, color in zip(sorted(scoring_regressor), ["g", "k"]):
    for sample, style in (("train", "--"), ("test", "-")):
        sample_score_mean = results_Elastic["mean_%s_%s" % (sample, scorer)]
        sample_score_std = results_Elastic["std_%s_%s" % (sample, scorer)]
        ax.fill_between(
            X_axis,
            sample_score_mean - sample_score_std,
            sample_score_mean + sample_score_std,
            alpha=0.1 if sample == "test" else 0,
            color=color,
        )
        ax.plot(
            X_axis,
            sample_score_mean,
            style,
            color=color,
            alpha=1 if sample == "test" else 0.7,
            label="%s (%s)" % (scorer, sample),
        )

    best_index = np.nonzero(results_Elastic["rank_test_%s" % scorer] == 1)[0][0]
    best_score = results_Elastic["mean_test_%s" % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot(
        [
            X_axis[best_index],
        ]
        * 2,
        [0, best_score],
        linestyle="-.",
        color=color,
        marker="x",
        markeredgewidth=3,
        ms=8,
    )

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()

##############################################################
#Definindo os modelos para o Ensemble
best_max_depth = 5
best_n_estimators = 20
best_n_neighbors = 49
best_C = 10
best_epsilon = 1
best_l1_ratio = 1

models = {
    'RF': RandomForestRegressor(max_depth = best_max_depth, n_estimators = best_n_estimators),
    'KNN': KNeighborsRegressor(n_neighbors = best_n_neighbors),
    'SVR': SVR(C = best_C, epsilon = best_epsilon),
    'EN': ElasticNetCV(l1_ratio = best_l1_ratio),
}

#Treinando e obtendo previsões
train_forecasts, test_forecasts = {}, {}
for k in models:
    models[k].fit(X_train, y_train)
    train_forecasts[k] = models[k].predict(X_train)
    test_forecasts[k] = models[k].predict(X_test)


#Reshape das previsões para o treino e teste
#test_forecasts['KNN'] = np.squeeze(test_forecasts['KNN'])
#test_forecasts['LASSO'] = np.squeeze(test_forecasts['LASSO'])
#test_forecasts['RF'] = np.squeeze(test_forecasts['RF'])
#test_forecasts['Ridge'] = np.squeeze(test_forecasts['Ridge'])
#test_forecasts['EN'] = np.squeeze(test_forecasts['EN'])
#print(test_forecasts['KNN'].ndim)

#train_forecasts['KNN'] = np.squeeze(train_forecasts['KNN'])
#train_forecasts['LASSO'] = np.squeeze(train_forecasts['LASSO'])
#train_forecasts['RF'] = np.squeeze(train_forecasts['RF'])
#train_forecasts['Ridge'] = np.squeeze(train_forecasts['Ridge'])
#train_forecasts['EN'] = np.squeeze(train_forecasts['EN'])

# previsões como pandas dataframe
ts_forecasts_df = pd.DataFrame(test_forecasts)
tr_forecasts_df = pd.DataFrame(train_forecasts)

# combining training and testing predictions
forecasts_df = pd.concat([tr_forecasts_df, ts_forecasts_df], axis=0).reset_index(drop=True)

# combining training and testing observations
actual = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

# setting up windowloss dynamic combinatio rule
windowing = WindowLoss()
window_weights = windowing.get_weights(forecasts_df, actual)
window_weights = window_weights.tail(X_test.shape[0]).reset_index(drop=True)

# setting up arbitrating dynamic combinatio rule
arbitrating = Arbitrating()
arbitrating.fit(tr_forecasts_df, y_train, X_train)
arb_weights = arbitrating.get_weights(X_test)
arb_weights = arb_weights.tail(X_test.shape[0])

# weighting the ensemble dynamically
windowing_fh = (window_weights.values * ts_forecasts_df.values).sum(axis=1)
arbitrating_fh = (arb_weights.values * ts_forecasts_df.values).sum(axis=1)
windowing_total = (window_weights.values * forecasts_df.values).sum(axis=1)

# combining the models with static and equal weights (average)
static_average = ts_forecasts_df.mean(axis=1).values

###############################################
#Obtenção das métricas de erro
#Quando a previsão é sobre o kt
windowing_fh = windowing_fh * test['I_cls_10']
#windowing_total = windowing_total * Dados['I_cls_10']
y_test = y_test * test['I_cls_10']
arbitrating_fh = arbitrating_fh * test['I_cls_10']
RF_fh = np.array(ts_forecasts_df[ts_forecasts_df.columns[0]])
RF_fh = RF_fh * test['I_cls_10']
KNN_fh = np.array(ts_forecasts_df[ts_forecasts_df.columns[1]])
KNN_fh = KNN_fh * test['I_cls_10']
SVR_fh = np.array(ts_forecasts_df[ts_forecasts_df.columns[2]])
SVR_fh = SVR_fh * test['I_cls_10']
EN_fh = np.array(ts_forecasts_df[ts_forecasts_df.columns[3]])
EN_fh = EN_fh * test['I_cls_10']

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, windowing_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, windowing_fh))
print("O valor de R² é:", r2_score(y_test, windowing_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, windowing_fh))

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, RF_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, RF_fh))
print("O valor de R² é:", r2_score(y_test, RF_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, RF_fh))

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, KNN_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, KNN_fh))
print("O valor de R² é:", r2_score(y_test, KNN_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, KNN_fh))

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, SVR_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, SVR_fh))
print("O valor de R² é:", r2_score(y_test, SVR_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, SVR_fh))

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, EN_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, EN_fh))
print("O valor de R² é:", r2_score(y_test, EN_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, EN_fh))

print("O valor de RMSE é:", np.sqrt(mean_squared_error(y_test, arbitrating_fh)))
print("O valor de MAE é:", mean_absolute_error(y_test, arbitrating_fh))
print("O valor de R² é:", r2_score(y_test, arbitrating_fh))
print("O valor de MAPE é:", mean_absolute_percentage_error(y_test, arbitrating_fh))

###############################################################
#Plotando alguns gráficos
#Distribuiçao dos pesos ao longo das previsões
plt.figure(figsize=(13, 13))
plt.xlabel("Test samples")
plt.ylabel("Windowing Weights")

# Escolhendo a palheta de cores com Seaborn.color_palette()
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
pal = sns.color_palette(palette='rocket', n_colors=4)

df_RF = np.zeros((4264, 2))
df_RF = pd.DataFrame(df_RF, columns=(['Weights', 'Models']))
df_RF['Models'] = df_RF['Models'].astype(str)
df_RF['Weights'] = window_weights['RF']
df_RF['Models'] = 'RF'

df_KNN = np.zeros((4264, 2))
df_KNN = pd.DataFrame(df_KNN, columns=(['Weights', 'Models']))
df_KNN['Models'] = df_KNN['Models'].astype(str)
df_KNN['Weights'] = window_weights['KNN']
df_KNN['Models'] = 'KNN'


df_SVR = np.zeros((4264, 2))
df_SVR = pd.DataFrame(df_SVR, columns=(['Weights', 'Models']))
df_SVR['Models'] = df_SVR['Models'].astype(str)
df_SVR['Weights'] = window_weights['SVR']
df_SVR['Models'] = 'SVR'

df_EN = np.zeros((4264, 2))
df_EN = pd.DataFrame(df_EN, columns=(['Weights', 'Models']))
df_EN['Models'] = df_EN['Models'].astype(str)
df_EN['Weights'] = window_weights['EN']
df_EN['Models'] = 'EN'

df = pd.concat([df_RF, df_KNN, df_SVR, df_EN], axis=0).reset_index(drop=True)

# Obtenção do valor médio dos pesos para cada modelo, adicionando uma nova couna ao dataframe
models_mean_serie = df.groupby('Models')['Weights'].mean()
df['mean_Weights'] = df['Models'].map(models_mean_serie)

#Utilizando o sns.FacetGrid
g = sns.FacetGrid(df, row='Models', hue='mean_Weights', aspect=15, height=0.75, palette=pal)

# Plotando as densidades para cada modelo utilizando kdeplot
g.map(sns.kdeplot, 'Weights',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)

# adicionamos uma linha branca representando o contorno de cada kdeplot
g.map(sns.kdeplot, 'Weights', 
      bw_adjust=1, clip_on=False, 
      color="w", lw=2)

## here we add a horizontal line for each plot
g.map(plt.axhline, y=0,
      lw=2, clip_on=False)

#Models dict
models_dict = {1: 'RF',
               2: 'KNN',
               3: 'SVR',
               4: 'EN'}


for i, ax in enumerate(g.axes.flat):
    ax.text(-15, 0.02, models_dict[i+1],
            fontweight='bold', fontsize=15,
            color=ax.lines[-1].get_color())
    
# we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
g.fig.subplots_adjust(hspace=-0.10)

# eventually we remove axes titles, yticks and spines
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
plt.xlabel('Weights Values', fontweight='bold', fontsize=15)
g.fig.suptitle('Weights Variation for models',
               ha='right',
               fontsize=20,
               fontweight=20)

plt.show()


#Radar Plot
radar = pd.DataFrame({
'Models': ['RF','KNN','SVR','EN', 'Windowing', 'Arbitrating'],
'RMSE': [80.68, 79.33, 78.31, 79.80, 77.47, 78.12],
'MAE': [51.86, 50.36, 44.86, 52.38, 47.32, 48.03],
'R²(%)': [91, 91, 91, 91, 91, 91],
'MAPE(%)': [21, 22, 19, 22, 20, 20]
})

#Parte 1: Definindo uma função para geração do radar plot
def make_spider( row, title, color):

    # Número de variáveis
    categories=list(radar)[1:]
    N = len(categories)

    # ângulo feito com cada eixo no plot (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Inicialize cada plot
    ax = plt.subplot(3,2,row+1, polar=True,)
    plt.subplots_adjust(hspace = 0.5)

    # Se quiser o primeiro eixo pode estar no topo:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # add eixo para cada variável + add labels labels
    plt.xticks(angles[:-1], categories, color='grey', size=8)

    # Fazendo as legendas do eixo y
    ax.set_rlabel_position(0)
    plt.yticks([10,20,30,40,50,60,70,80,90],["10","20","30","40","50","60","70","80","90"], color="grey", size=7)
    plt.ylim(0,100)

    # Ind1
    values=radar.loc[row].drop('Models').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add um título
    plt.title(title, size=11, color=color, y=1.1)


#Parte 2: Aplicando a função para cada modelo
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
# Criando a palheta de cores:
my_palette = plt.cm.get_cmap("Set2", len(radar.index))
 
# Loop to plot
for row in range(0, len(radar.index)):
    make_spider( row=row, title= radar['Models'][row], color=my_palette(row))

#Plots para regressão
y = Dados['GHI_10']
df_reg = df = pd.concat([windowing_total, Dados['GHI_10']], axis=1).reset_index(drop=True)
df_reg['split'] = 'train'
df_reg.loc[np.array(test_idx), 'split'] = 'test'
fig = px.scatter(x=y_test, y=windowing_fh, labels={'x': 'ground truth', 'y': 'prediction'}, template='plotly_white')
fig.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max()
)
fig.show()


fig = px.scatter(
    df_reg, x=y, y=windowing_total,
    marginal_x='histogram', marginal_y='histogram',
    color='split', trendline='ols', template='plotly_white', opacity=0.5
)

fig.update_traces(histnorm='probability', selector={'type':'histogram'})
fig.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y.min(), y0=y.min(),
    x1=y.max(), y1=y.max()
)

fig.show()

#plots para residuos
df_reg['residual'] = df_reg['GHI_10'] - df_reg['I_cls_10']
df_reg_nna = df_reg.dropna()

fig = px.scatter(
    df_reg_nna, x='I_cls_10', y='residual',
    marginal_y='violin',
    color='split', trendline='ols', template='plotly_white'
)
fig.show()
#########################################################
