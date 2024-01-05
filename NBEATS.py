'''
Implementação do modelo desafiador N-BEATS
Autor: Felipe Pinto Marinho
Data:05/01/2024
'''

#Carregando alguns pacotes relevantes
import pandas as pd
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Carregando os dados
data = pd.read_csv('train.csv', index_col='id', parse_dates=['date'])
data2 = data.loc[(data['store_nbr'] == 1) & (data['family'].isin(['MEATS', 'PERSONAL CARE'])), ['date', 'family', 'sales', 'onpromotion']]

#Adcionando o mês de dezembro
dec25 = list()
for year in range(2013,2017):
    for family in ['MEATS', 'PERSONAL CARE']:
        dec18 = data2.loc[(data2['date'] == f'{year}-12-18') & (data2['family'] == family)]
        dec25 += [{'date': pd.Timestamp(f'{year}-12-25'), 'family': family, 'sales': dec18['sales'].values[0], 'onpromotion': dec18['onpromotion'].values[0]}]
data2 = pd.concat([data2, pd.DataFrame(dec25)], ignore_index=True).sort_values('date')

#Codificando as colunas para ficarem com a terminologia adequada para a entrada nas funções do pacote
data2 = data2.rename(columns={'date': 'ds', 'sales': 'y', 'family': 'unique_id'})
data2.head() 

#Divisão terino/teste temporal
train = data2.loc[data2['ds'] < '2017-01-01']
valid = data2.loc[(data2['ds'] >= '2017-01-01') & (data2['ds'] < '2017-04-01')]
h = valid['ds'].nunique()

#Definição, treinamento e previsão com o NBEATS
models = [NBEATS(h=h,input_size=7,
                 loss=DistributionLoss(distribution='Poisson', level=[90]),
                 max_steps=100,
                 scaler_type='standard',
               futr_exog_list=['onpromotion'])]

model = NeuralForecast(models=models, freq='D')
model.fit(train)

p =  model.predict(futr_df=valid).reset_index()
p = p.merge(valid[['ds','unique_id', 'y']], on=['ds', 'unique_id'], how='left')
p.head()

#Otimização Bayesiana para tunning dos hiperparâmetros do NBEATS
def objective(trial):
    input_size = trial.suggest_int('input_size', 1, 60)
    
    n_blocks_season = trial.suggest_int('n_blocks_season', 1, 3)
    n_blocks_trend = trial.suggest_int('n_blocks_trend', 1, 3)
    n_blocks_identity = trial.suggest_int('n_blocks_ident', 1, 3)
    
    mlp_units_n = trial.suggest_categorical('mlp_units', [32, 64, 128])
    num_hidden = trial.suggest_int('num_hidden', 1, 3)
    
    n_harmonics = trial.suggest_int('n_harmonics', 1, 5)
    n_polynomials = trial.suggest_int('n_polynomials', 1, 5)
    
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    
    n_blocks = [n_blocks_season, n_blocks_trend, n_blocks_identity]
    mlp_units=[[mlp_units_n, mlp_units_n]]*num_hidden
    models = [NBEATS(h=h,input_size=input_size,
                 loss=DistributionLoss(distribution='Poisson', level=[90]),
                 max_steps=100,
                 futr_exog_list=['onpromotion'],
                 stack_types=['seasonality', 'trend', 'identity'],
                 mlp_units=mlp_units,
                 n_blocks=n_blocks,
                 learning_rate=learning_rate,
                 n_harmonics=n_harmonics,
                 n_polynomials=n_polynomials,
                 scaler_type=scaler_type)
                 ]
    model = NeuralForecast(models=models, freq='D')
    model.fit(train)

    p = model.predict(futr_df=valid).reset_index()
    p = p.merge(valid[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')

    loss = mean_absolute_error(p['y'], p['NBEATS']) 

    return loss

#Executando a otimização usando pacote optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

study.best_params
study.best_value
