# -*- coding: utf-8 -*-
"""
Implementação da metodologia de Group Method Data Handling (GMDH)
Autor:Felipe Pinto Marinho
Data: 05/01/2024
"""

#Carregando alguns pacotes relevantes
import numpy as np
import matplotlib.pyplot as plt
from gmdh import Combi, split_data

#Criação de dataset arbitrário
X = [[1, 2], [3, 2], [7, 0], [5, 5], [1, 4], [2, 6]]
y = [3, 5, 7, 10, 5, 8]

#Divisão treino/teste
x_train, x_test, y_train, y_test = split_data(X, y)

print('x_train:\n', x_train)
print('x_test:\n', x_test)
print('\ny_train:\n', y_train)
print('y_test:\n', y_test)

#Ajuste do modelo no treino e predição no teste
model = Combi()
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

#Comparando valores predito com real
print('y_predicted: ', y_predicted)
print('y_test: ', y_test)

#Polinômio ótimo
model.get_best_polynomial()
