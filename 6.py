from utils.neuron import Neuron, step_function
import numpy as np


# 6. Explique cómo combinan estas neuronas las diferentes entradas que reciben.

# El perceptron utiliza matrices de columna para operar entre las distintas entradas que recibe de nuestro programa.

# Los genera utilizando numeros aleatorios con np.random

# Y luego, la funcion forward de la clase Neuron es la que procesa las entradas genera una recta de regresión lineal y los pasa por nuestra función de activación
