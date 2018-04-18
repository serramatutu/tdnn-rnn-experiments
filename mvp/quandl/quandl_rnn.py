import tensorflow as tf
import random
import time
import datetime

# para uso do dataset quandl.
import sys
sys.path.append('../../datasets/quandl')

from read_tfrecord import read_multiple

input_size = 2 # data e dia

rnn = tf.contrib.rnn

# HYPERPARAMS
learning_rate = 0.001  # taxa de treinamento
training_iters = 50000 # quantas iterações de treinamento
n_epochs = 1           # quantas vezes passa por toda a sequência
display_step = 1000    # de quanto em quanto tempo ativar o logger
n_input = 50           # quanto a RNN vai olhar para trás
n_hidden = 512         # número de células LSTM por camada
depth = 2              # profundidade das camadas

# placeholders de entrada e saída da RNN
batch = tf.placeholder(tf.float32, [n_input, 2], "batch")

x = batch # input é o mesmo que a batch
y = batch[:, 0] # a saída é apenas o valor de predição (indice 0 da linha)

# inicializa os pesos e viézes
weights = {
    "out": tf.Variable(tf.random_normal([n_hidden, 1]))
}
biases = {
    "out": tf.Variable(tf.random_normal([1]))
}

# a função RNN gera uma predição baseada no input (x), nos pesos (w) e nas biases (b)
def RNN(x, w, b):
    # flatten de input para ficar do tamanho esperado e nas dimensoes esperadas [1xn]
    # x = tf.reshape(x, [-1, n_input])
    # print(x)

    # quebra o tensor da batch em vários inputs de tamanho n_input
    x = tf.split(x, n_input, 0)

    # camadas LSTM de largura n_hidden e profundidade depth
    cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden) for i in range(depth)])

    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # multiplica o último output (predição) pelos pesos soma a bias
    return tf.matmul(outputs[-1], w['out']) + b['out']

pred = RNN(x, weights, biases)
pred = pred[0, 0]

# função de perda ou custo (loss function)
# y[-1] pois só estamos interessados no último output
cost = tf.squared_difference(pred, y[-1])

# otimizador (que corrigirá o erro da RNN)
# TODO: estudar isso depois
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Avaliar a precisão
# TODO: ver no que dá :p
difference = tf.abs(pred - y) # calcula as diferenças entre valores previstos e valores esperados
accuracy = tf.reduce_mean(difference/y) # calcula a média das porcentagens de erro


init = tf.global_variables_initializer()

# LOGGING
start_time = time.time()
def elapsed(sec):
    return str(datetime.timedelta(seconds=sec))

logs_path = '/tmp/tensorflow/rnn_oil'
writer = tf.summary.FileWriter(logs_path)

def print_step(step, accuracy_total, loss_total):
    if (step+1) % display_step == 0:
        print("Iteration: " + str(step+1) + " (" + str(step/training_iters) + "%)")
        print("Average Loss: " + "{:.6f}".format(loss_total/display_step))
        print("Average Accuracy: " + "{:.2f}%".format(100*accuracy_total/display_step))


with tf.Session as session:
    session.run(init) # inicializa o grafo

    # para logging
    accuracy_total = 0
    loss_total = 0

    # salva o grafo atual
    writer.add_graph(session.graph)

    dataset = read_multiple("data.tfrecord")
    batches = dataset.apply(tf.contrib.data.batch_and_drop_remainder(n_input))  
    batches.shuffle(buffer_size=10000) # aleatoriza as batches

    iterator = batches.make_initializable_iterator()
    next_batch = iterator.get_next()

    step = 0
    for i in range(n_epochs):
        if step >= training_iters:
            break

        session.run(iterator.initializer)

        while step < training_iters:
            value = session.run(next_batch)

            try:
                _, accuracy, loss, predition = session.run([optimizer, accuracy, cost, pred], 
                                                            feed_dict = {
                                                                batch: next_batch,
                                                            })
                step += 1
                print_step(step, accuracy_total, loss_total)
            except tf.errors.OutOfRangeError: # quando terminou as batches desta epoch
                break

        print('Epoch '+i+' finished training.')