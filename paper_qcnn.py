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
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
import pylatexenc

#Para o NBEATS
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import DistributionLoss
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Para o GMDH
import pandas as pd
import seaborn as sns
from gmdh import Combi, split_data
from scipy.io import loadmat

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
GHI/Para 5 min a posteriori/Features e Targets
'''
features_irradiance_5_min = features_irradiance[['B(ghi_kt|5min)', 'V(ghi_kt|5min)', 'L(ghi_kt|5min)']]
features_sky_images = features_sky_images.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)'])
features_5_min = pd.concat([features_irradiance_5_min, features_sky_images], axis = 1) #features irradiance + features sky images
list(features_5_min.columns.values)
features_5_min.head()
features_5_min.tail()

target_5_min = target_intra_hour[['ghi_5min', 'ghi_clear_5min', 'ghi_kt_5min', 'timestamp']]
list(target_5_min.columns.values)
target_5_min.head()
target_5_min.tail()

'''
Etapa de pré-processamento
Remoção de NA's, filtros de volumetria e volatilidade
'''
