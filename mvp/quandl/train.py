import numpy as np
import tensorflow as tf
import random
import time
import datetime

# HYPERPARAMS
learning_rate = 0.001  # taxa de treinamento
training_iters = 50000 # quantas iterações de treinamento
n_epochs = 15          # quantas vezes passa por toda a sequência
display_step = 50    # de quanto em quanto tempo ativar o logger
n_input = 200          # quanto a RNN vai olhar para trás
n_hidden = 512         # número de células LSTM por camada
depth = 2              # profundidade das camadas

FLAGS = None

def get_dataset(dataset_path):
    # TODO: fazer um arquivo só para os dados
    return read_multiple(dataset_path)

def get_batch_iterator(dataset_path):
    batches = get_dataset(dataset_path).apply(tf.contrib.data.batch_and_drop_remainder(n_input))  
    batches.shuffle(buffer_size=10000) # aleatoriza as batches

    return batches.make_initializable_iterator()

def get_batch_placeholders()

def print_step(step, accuracy_total, loss_total):
    if (step+1) % display_step == 0:
        print("Iteration: " + str(step+1) + " (" + str(step/training_iters) + "%)")
        print("Average Loss: " + "{:.6f}".format(loss_total/display_step))
        print("Average Accuracy: " + "{:.2f}%".format(100*accuracy_total/display_step))
        return True
    return False

def elapsed(sec):
    return str(datetime.timedelta(seconds=sec))

def train():
    """Treina a RNN"""
    start_time = time.time()

    # iterador que percorre o arquivo de dados
    iterator = get_batch_iterator(FLAGS.dataset_path)
    next_batch = iterator.get_next()

    # logger
    writer = tf.summary.FileWriter(FLAGS.log_path)

    init = tf.global_variables_initializer()

    with tf.graph().as_default():
        batch
        session.run(init) # inicializa o grafo

        # para logging
        accuracy_total = 0
        loss_total = 0

        # salva o grafo atual
        writer.add_graph(session.graph)

        step = 0
        for i in range(n_epochs):
            if step >= training_iters:
                break

            session.run(iterator.initializer)

            while step < training_iters:
                try:
                    batch_value = session.run(next_batch)
                    batch_value = np.array([batch_value[0], batch_value[1]])
                    batch_value = batch_value.transpose()[0]
                    _, acc, loss, prediction = session.run([optimizer, accuracy, cost, pred], 
                                                                feed_dict = {
                                                                    batch: batch_value,
                                                                })
                    accuracy_total += acc
                    loss_total += loss
                    step += 1

                    if print_step(step, accuracy_total, loss_total):
                        accuracy_total = 0
                        loss_total = 0
                except tf.errors.OutOfRangeError: # quando terminou as batches desta epoch
                    break

            print('Epoch '+str(i)+' finished training.')