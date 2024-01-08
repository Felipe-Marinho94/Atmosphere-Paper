# -*- coding: utf-8 -*-
"""
Implementação de uma rede QNN para problemas de regressão
Autor:Felipe Pinto Marinho
Data: 08/01/2024 
"""

#Importando alguns pacotes relevantes
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

algorithm_globals.random_seed = 42

#Função que retorna o valor da função custo na estapa de treinamento
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

#Criando um dataset sintético para trabalhar com problemas de regressão
num_samples = 20
eps = 0.2
lb, ub = -np.pi, np.pi
X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
f = lambda x: np.sin(x)

X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

plt.plot(X_, f(X_), "r--")
plt.plot(X, y, "bo")
plt.show()

# Construção de um simples feature map
param_x = Parameter("x")
feature_map = QuantumCircuit(1, name="fm")
feature_map.ry(param_x, 0)

# cConstrução de um simples ansatz
param_y = Parameter("y")
ansatz = QuantumCircuit(1, name="vf")
ansatz.ry(param_y, 0)

# Construção do circuito quântico
qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)

# Construção da QNN
regression_estimator_qnn = EstimatorQNN(circuit=qc)

# Construção do regressor  para a rede neural
regressor = NeuralNetworkRegressor(
    neural_network=regression_estimator_qnn,
    loss="squared_error",
    optimizer=L_BFGS_B(maxiter=500),
    callback=callback_graph,
)

# Criação de array vazio para armazenar os valores da função de custo
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

# Ajustando nos dados de treino
regressor.fit(X, y)

# Retornando o valor default para o tamanho da figura
plt.rcParams["figure.figsize"] = (6, 4)

# Resultado para a métrica de desempenho
regressor.score(X, y)

# plotando a função alvo
plt.plot(X_, f(X_), "r--")

# plotando os dados
plt.plot(X, y, "bo")

# plotando a linha de ajuste
y_ = regressor.predict(X_)
plt.plot(X_, y_, "g-")
plt.show()

regressor.weights

#Regressão com regressor variacional quântico
vqr = VQR(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=L_BFGS_B(maxiter=5),
    callback=callback_graph,
)

#Otimização do VQR
#Criando array vazio para armazenar os valores da função custo
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

# fit regressor
vqr.fit(X, y)

# return to default figsize
plt.rcParams["figure.figsize"] = (6, 4)

# score result
vqr.score(X, y)
