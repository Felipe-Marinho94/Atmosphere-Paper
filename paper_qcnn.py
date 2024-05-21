# -*- coding: utf-8 -*-
"""
Trabalhando com o conjuntode dados de Folsom para aplicação dos modelos QCNN, NBEATS e GMDH
Autor:Felipe Pinto Marinho
Data:08/01/2024
"""

#Carregando alguns pacotes relevantes
#Para o QCNN
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap, TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
import pylatexenc
import pennylane as qml
from sklearn.svm import SVR
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_algorithms.utils import algorithm_globals

#Para ML
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

#Para o GMDH
import pandas as pd
import seaborn as sns
from gmdh import Combi, split_data

#LightGBM
import lightgbm as lgb
import optuna as opt

#XGBoost
from xgboost import XGBRegressor
from xgboost import plot_importance

#Carregando o dataset para análise
features_irradiance = pd.read_csv("Irradiance_features_intra-hour.csv", sep = ",")
features_sky_images = pd.read_csv("Sky_image_features_intra-hour.csv", sep = ",")
target_intra_hour =  pd.read_csv("Target_intra-hour.csv", sep = ",")

#Pré-visualização dos dataframes
features_irradiance.head()
features_irradiance.tail()

features_sky_images.head()
features_sky_images.tail()

target_intra_hour.head()
target_intra_hour.tail()

#Avaliando o que cada coluna representa
list(features_irradiance.columns.values)
list(features_sky_images.columns.values)
list(target_intra_hour.columns.values)

'''
Gerando os datasets para cada horizonte de previsão
GHI/Para 30 min a posteriori/Features e Targets
'''
features_irradiance_30_min = features_irradiance[['B(dni_kt|30min)', 'V(dni_kt|30min)', 'L(dni_kt|30min)']]
features_sky_images = features_sky_images.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)'])
features_30_min = pd.concat([features_irradiance_30_min, features_sky_images], axis = 1) #features irradiance + features sky images
list(features_30_min.columns.values)
features_30_min.head()
features_30_min.tail()

target_30_min = target_intra_hour[['dni_30min', 'dni_clear_30min', 'dni_kt_30min', 'timestamp']]
list(target_30_min.columns.values)
target_30_min.head()
target_30_min.tail()

'''
Etapa de pré-processamento
Remoção de NA's, filtros de volumetria e volatilidade
'''
df = pd.concat([features_30_min.drop(columns = ['timestamp']), target_30_min], axis=1)
df_sem_Na = df.dropna()
list(df_sem_Na.columns.values)

'''
Divisão treino/teste temporal
Será utilizados os anos de 2013,2014,2015 para treino dos modelos
2016 será utilizado para teste
'''
train = df_sem_Na.loc[df_sem_Na['timestamp'] < '2016-01-01']
test = df_sem_Na.loc[(df_sem_Na['timestamp'] >= '2016-01-01') & (df_sem_Na['timestamp'] < '2016-12-31')]
features_train = train.drop(columns = ['dni_30min','dni_clear_30min','dni_kt_30min', 'timestamp'])
target_train = train[['dni_30min','dni_clear_30min','dni_kt_30min']]
features_test = test.drop(columns = ['dni_30min','dni_clear_30min','dni_kt_30min', 'timestamp'])
target_test = test[['dni_30min','dni_clear_30min','dni_kt_30min']]

#------------------------------------------------------------------------------
#Implementação do modelo de Group Method Data Handling (GMDH)
#Fundamentado pela documentação do pacote GMDH: https://gmdh.net/index.html
#------------------------------------------------------------------------------
model_GMDH = Combi()
model_GMDH.fit(np.array(features_train), np.array(target_train['dni_kt_30min']))
GMDH_hat_30min = model_GMDH.predict(features_test)

#Conversão em inrradiância utilizando a irradiância de céu claro
GMDH_hat_30min = GMDH_hat_30min * target_test['dni_clear_30min']

#Avaliação do desempenho: RMSE, MAE, R², MAPE
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['dni_30min'], GMDH_hat_30min)))
print("O valor do MAE é:", mean_absolute_error(target_test['dni_30min'], GMDH_hat_30min))
print("O valor do R² é:", r2_score(target_test['dni_30min'], GMDH_hat_30min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['dni_30min'], GMDH_hat_30min))

#------------------------------------------------------------------------------
#Implementação do modelo Light Gradient Boosting Method (lightGBM)
#Fundamentado pela documentação do pacote lightGBM: https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
#------------------------------------------------------------------------------
#Carregando os dados para o modelo lightGBM
train_lightgbm = lgb.Dataset(features_train, target_train['dni_kt_30min'])
test_lightgbm = lgb.Dataset(features_test, target_test['dni_kt_30min'], reference = train_lightgbm)

#Configuração dos parâmetros
params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learning_rate': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1
}

#Ajustando o modelo
model = lgb.train(params,
                 train_set = train_lightgbm,
                 valid_sets = test_lightgbm)

#Previsão
lightGBM_hat_30min = model.predict(features_test)

#Conversão para irradiância
lightGBM_hat_30min = lightGBM_hat_30min * target_test['dni_clear_30min']

#Avaliação do Desempenho
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['dni_30min'], lightGBM_hat_30min)))
print("O valor do MAE é:", mean_absolute_error(target_test['dni_30min'], lightGBM_hat_30min))
print("O valor do R² é:", r2_score(target_test['dni_30min'], lightGBM_hat_30min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['dni_30min'], lightGBM_hat_30min))

#Importância de Atributos
lgb.plot_importance(model, figsize=(7,6), title="LightGBM Feature Importance")
plt.show()

#Procedimento de Tunning, para referência: https://forecastegy.com/posts/how-to-use-optuna-to-tune-lightgbm-hyperparameters/
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "verbosity": -1,
        "bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(features_train, target_train['dni_kt_30min'])
    predictions = model.predict(features_test)
    rmse = mean_squared_error(target_test['dni_kt_30min'], predictions, squared=True)
    return rmse

#Otimização utilizando o pacote Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)

#Utilizando o LGBM otimizado
model = lgb.train(study.best_params,
                 train_set = train_lightgbm,
                 valid_sets = test_lightgbm)

#Previsão
lightGBM_hat_30min = model.predict(features_test)

#Conversão para irradiância
lightGBM_hat_30min = lightGBM_hat_30min * target_test['ghi_clear_30min']

#Avaliação do Desempenho
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['ghi_30min'], lightGBM_hat_30min)))
print("O valor do MAE é:", mean_absolute_error(target_test['ghi_30min'], lightGBM_hat_30min))
print("O valor do R² é:", r2_score(target_test['ghi_30min'], lightGBM_hat_30min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['ghi_30min'], lightGBM_hat_30min))

#------------------------------------------------------------------------------
#Implementação do modelo Quantum Support Vector Regression (QSVR)
#Fundamentado pela documentação do pacote QISKIT: 
#------------------------------------------------------------------------------
nqubits = 1
dev = qml.device("lightning.qubit", wires = nqubits)

@qml.qnode(dev)
def kernel_circ(a, b):
    qml.AmplitudeEmbedding(
        a, wires = range(nqubits), pad_with = 0, normalize = True)
    qml.adjoint(qml.AmplitudeEmbedding(
        b, wires = range(nqubits), pad_with = 0, normalize = True))
    return qml.probs(wires=range(nqubits))

#Realizando alguns testes
kernel_circ(np.array(features_train)[0], np.array(features_train)[0])[0]

#Utilizando o kernel quântico desenvolvido para um SVC do sklearn
def qkernel(A, B):
    return np.array([[kernel_circ(a, b)[0] for b in B] for a in A])

np.array(features_train)[0:5000, :]
np.array(target_train['ghi_kt_5min'])[0:100]
qsvr = QSVR(kernel = qkernel).fit(np.array(features_train)[0:500, :],
                                 np.array(target_train['ghi_kt_5min'])[0:500])

QSVR_hat_5min = qsvr.predict(np.array(features_test)[0:100, :],)
QSVR_hat_5min = QSVR_hat_5min * target_test['ghi_clear_5min'][0:100]

qsvr = QSVR()
qsvr.fit(np.array(features_train)[0:100, :],
        np.array(target_train['ghi_kt_5min'])[0:100])

print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['ghi_5min'][0:100], QSVR_hat_5min)))
print("O valor do MAE é:", mean_absolute_error(target_test['ghi_5min'][0:100], QSVR_hat_5min))
print("O valor do R² é:", r2_score(target_test['ghi_5min'][0:100], QSVR_hat_5min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['ghi_5min'][0:100], QSVR_hat_5min))

#------------------------------------------------------------------------------
#Implementação da seleção de features utilizando Recursive Features Elimination]
#RFE - source:https://machinelearningmastery.com/rfe-feature-selection-in-python/ 
#------------------------------------------------------------------------------
rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=1)
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

#Avaliação do modelo
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
n_scores = cross_val_score(pipeline, features_train, target_train['ghi_kt_10min'],
                           scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

#Reportando desempenho
print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#Selecionando a features
rfe.fit(features_train, target_train['ghi_kt_10min'])

#selecionando as features
for i in range(features_train.shape[1]):
 print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
 
list(features_train.columns)

#------------------------------------------------------------------------------
#Implementação do Variational Quantum Regressor (VQR)
#VQR - source:https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html 
#------------------------------------------------------------------------------
#Construindo um fature map simples
X_train = np.array(features_train['B(ghi_kt|10min)']).reshape(len(features_train), 1)
y_train = np.array(target_train['ghi_kt_10min'])

X_test = np.array(features_test['B(ghi_kt|10min)']).reshape(len(features_test), 1)
y_test = np.array(target_test['ghi_kt_10min'])

#Normalizando os dados
scale = StandardScaler()
X_train_scale = scale.fit_transform(X_train) 
X_test_scale = scale.fit_transform(X_test)

param_x = Parameter("x")
feature_map = QuantumCircuit(len(X_train_scale[0]), name="fm")
params_x = [Parameter(f"x{i+1}") for i in range(len(X_train_scale[0]))]

#angle encoding
for i, param in enumerate(params_x):
     feature_map.ry(param, i)

#param_x = Parameter("x")
#feature_map = QuantumCircuit(1, name="fm")
#feature_map.ry(param_x, 0)

ansatz = TwoLocal(1, 'ry', 'cx', 'linear', reps = 1)

#Construindo um ansatz simples
#param_y = Parameter("y")
#ansatz = QuantumCircuit(1, name="vf")
#ansatz.ry(param_y, 0)

#Construindo um circuito
qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)

#Construindo a QNN
regression_estimator_qnn = EstimatorQNN(circuit=qc)

objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

regressor = NeuralNetworkRegressor(
    neural_network=regression_estimator_qnn,
    loss="squared_error",
    optimizer=L_BFGS_B(maxiter=5),
    callback=callback_graph,
)

plt.rcParams["figure.figsize"] = (12, 6)

#X_train[:100].shape
#np.array(y_train[:100])

regressor.fit(X_train_scale, y_train)
VQR_hat_10min = regressor.predict(X_test_scale)
VQR_hat_10min = np.squeeze(VQR_hat_10min)

VQR_hat_10min = VQR_hat_10min * target_test['ghi_clear_10min']
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['ghi_10min'], VQR_hat_10min)))
print("O valor do MAE é:", mean_absolute_error(target_test['ghi_10min'], VQR_hat_10min))
print("O valor do R² é:", r2_score(target_test['ghi_10min'], VQR_hat_10min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['ghi_10min'], VQR_hat_10min))

#------------------------------------------------------------------------------
#Implementação do modelo XGBoost
#source:
#------------------------------------------------------------------------------
#Incializando o modelo
XGBoost = XGBRegressor()

#Definindo o método de avaliação do modelo
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)

scores = cross_val_score(XGBoost, features_train, target_train['ghi_kt_15min'],
                         scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

#score positivo
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


#Tunning do modelo XGBoost utilizando otimização Bayesiana (Optuna)
#source:https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/
def objective(trial):
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    model = XGBRegressor(**params)
    model.fit(features_train, target_train['ghi_kt_30min'], verbose=False)
    predictions = model.predict(features_test)
    rmse = mean_squared_error(target_test['ghi_kt_30min'], predictions, squared=False)
    return rmse

#realizando a otimização
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)
best_params = {
    "objective": "reg:squarederror",
    "n_estimators": 1000,
    "verbosity": 0,
    "learning_rate": 0.007862369974562914,
    "max_depth":  6,
    "subsample": 0.6476886673566187,
    "colsample_bytree": 0.8835466631816744,
    "min_child_weight": 12,
}

#Realizando previsões
XGBoost = XGBRegressor(**best_params)
XGBoost.fit(features_train, target_train['ghi_kt_30min'])
XGBoost_hat_30min = XGBoost.predict(features_test)
XGBoost_hat_30min = XGBoost_hat_30min * target_test['ghi_clear_30min']
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['ghi_30min'], XGBoost_hat_30min)))
print("O valor do MAE é:", mean_absolute_error(target_test['ghi_30min'], XGBoost_hat_30min))
print("O valor do R² é:", r2_score(target_test['ghi_30min'], XGBoost_hat_30min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['ghi_30min'], XGBoost_hat_30min))

plot_importance(XGBoost, title = "XGBoost Feature Importance")
