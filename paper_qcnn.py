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
features_irradiance = pd.read_csv("Folsom_irradiance.csv", sep = ",")
features_sky_images = pd.read_csv("Folsom_sky_image_features.csv", sep = ",")
target_intra_hour =  pd.read_csv("Target_intra-hour.csv", sep = ",")

#Pré-visualização dos dataframes
features_irradiance.head()
features_sky_images.head()
target_intra_hour.head()
