import numpy as np
import tensorflow as tf
import random
import time
import datetime

# para uso do dataset quandl.
import sys
sys.path.append('../../datasets/quandl')

rnn = tf.contrib.rnn

# a função RNN gera uma predição baseada no input (x), nos pesos (w) e nas biases (b)
def predict(batch, n_units, depth):
    """Constrói o modelo da RNN para previsão

    Args:
        batch: um placeholder que contém uma batch de dados a processar, com shape (?, 2)
        n_units: o número de units nas células LSTM
        depth: profundidade da rede

    Returns:
        prediction: um Tensor com shape (), contendo a predição final para a sequência dada

    """
    # quebra o tensor da batch em vários inputs de tamanho n_input (batch.shape[0])
    x = tf.split(batch, batch.shape[0], 0)

    # camadas LSTM de largura n_hidden e profundidade depth
    cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_units) for i in range(depth)])

    # cria a rede em si e processa a partir da célula LSTM e dos dados e entrada
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # pra ficar organizadinho :)
    with tf.name_scope("rnn"):
        # inicializa os pesos e viézes
        weights =  tf.Variable(tf.random_normal([n_units, 1]), name="weights")
        bias = tf.Variable(tf.random_normal([1]), name="bias")

        # multiplica o último output (predição) pelos pesos soma a bias
        prediction = tf.matmul(outputs[-1], weights) + bias
        prediction = prediction[0, 0]

    return prediction

def loss(pred, y):
    """Calcula a perda (erro) da previsão

    Args:
        pred: a previsão dada pela rede
        y: o valor esperado (deve ser o último da batch (y[-1]))

    Returns:
        loss: o erro da previsão
    """
    loss = tf.squared_difference(pred, y)
    return loss

def train(loss, learning_rate):
    """Treina a RNN baseada no erro dela e na taxa de treinamento

    Cria um summarizer para acompanhar a perda ao longo do tempo no TensorBoard

    Acompanha o step global de treinamento através de uma tf.Variable

    Args:
        loss: o erro da previsão calculado por loss()
        learning_rate: a taxa de aprendizado

    Returns:
        train_op: uma operação que representa o treinamento da rede
    """
    # escreve no Summary do TensorBoard para acompanhamento do aprendizado
    tf.summary.scalar('loss', loss)

    # step não é treinável pois não varia de acordo com o otimizador
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # otimizador (que corrigirá o erro da RNN)
    # TODO: estudar isso depois
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    return train_op


def evaluation(pred, y):
    """Avalia a precisão ou a "qualidade" de um modelo baseado na previsão e no valor esperado

    Args:
        pred: a previsão dada pela rede
        y: o valor esperado (deve ser o último da batch (y[-1]))

    Returns:
        accuracy: um tensor escalar de tipo tf.float32 com a precisão do modelo 
    """
    # Avaliar a precisão
    # TODO: ver no que dá :p
    difference = tf.abs(pred - y) # calcula as diferenças entre valores previstos e valores esperados
    accuracy = tf.reduce_mean(difference/y) # calcula a média das porcentagens de erro
    return accuracy