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
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
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

#Função para avaliação da função custo na etapa de treinamento
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

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
df = pd.concat([features_5_min.drop(columns = ['timestamp']), target_5_min], axis=1)
df_sem_Na = df.dropna()
list(df_sem_Na.columns.values)

'''
Divisão treino/teste temporal
Será utilizados os anos de 2013,2014,2015 para treino dos modelos
2016 será utilizado para teste
'''
train = df_sem_Na.loc[df_sem_Na['timestamp'] < '2016-01-01']
test = df_sem_Na.loc[(df_sem_Na['timestamp'] >= '2016-01-01') & (df_sem_Na['timestamp'] < '2017-12-31')]
features_train = train.drop(columns = ['ghi_5min','ghi_clear_5min','ghi_kt_5min', 'timestamp'])
target_train = train[['ghi_5min','ghi_clear_5min','ghi_kt_5min']]

'''
CNN quântica para previsão de irradiância solar de curto prazo
'''

algorithm_globals.random_seed = 12345

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target


# Desenhando o circuito quântico para os dados de entrada
params = ParameterVector("θ", length=3)
circuit = conv_circuit(params)
circuit.draw("mpl")

#Função para implementar a camada convolucional quântica
def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


circuit = conv_layer(4, "θ")

#Desenhando a camada convolucional quântica
circuit.decompose().draw("mpl")

#Criação da camada de poooling
def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target


params = ParameterVector("θ", length=3)
circuit = pool_circuit(params)
circuit.draw("mpl")

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


sources = [0, 1]
sinks = [2, 3]
circuit = pool_layer(sources, sinks, "θ")
circuit.decompose().draw("mpl")

#Codificação do dataset em um circuito quântico com 8 qubits utilizando ZFeatureMap
feature_map = ZFeatureMap(15)
feature_map.decompose().draw("mpl")

#Treinando a rede QCNN, utilizando como porta quântica para a medição a porta de Pauli esparsa
feature_map = ZFeatureMap(15)

ansatz = QuantumCircuit(15, name="Ansatz")

#Construção da QCNN
# Primeira Camada Convolucional
ansatz.compose(conv_layer(15, "с1"), list(range(15)), inplace=True)

# Primeira Camada de Pooling
ansatz.compose(pool_layer([0, 1, 2, 3, 4, 5, 6, 7], 
                          [8, 9, 10, 11, 12, 13, 14], "p1"),
               list(range(15)), inplace=True)

# Segunda Camada Convolucional
ansatz.compose(conv_layer(4, "c2"), list(range(11, 15)), inplace=True)

# Segunda Camada de Pooling
ansatz.compose(pool_layer([0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10], "p2"),
               list(range(4, 15)), inplace=True)

# Terceira Camada Convolucional
ansatz.compose(conv_layer(9, "c3"), list(range(6, 15)), inplace=True)

# Terceira Camada de Pooling
ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

# Combinando o feature map e ansatz
circuit = QuantumCircuit(15)
circuit.compose(feature_map, range(15), inplace=True)
circuit.compose(ansatz, range(15), inplace=True)

# construindo o regressor para a rede
regression_estimator_qnn = EstimatorQNN(
    circuit=circuit.decompose(),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)


circuit.draw("mpl")

regressor = NeuralNetworkRegressor(
    neural_network=regression_estimator_qnn,
    loss="squared_error",
    optimizer=L_BFGS_B(maxiter=5),
    callback=callback_graph,
)

#Treinamento da rede
x = np.asarray(features_train)
y = np.asarray(target_train[['ghi_kt_5min']])
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)
regressor.fit(x, y)
