#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random
import time
import datetime
import argparse
import sys
import os

import data
import model
import evaluation

FLAGS = None

def make_feed_dict(batch, x_placeholder, y_placeholder):
    return {
        x_placeholder: batch,
        y_placeholder: batch[1, -1]
    }

def train():
    """Treina a RNN"""
    start_time = time.time()

    # dataset com os dados de treinamento
    dataset = data.QuandlDataset(FLAGS.dataset_path)

    with tf.Graph().as_default():
        # cria o modelo
        # x tem tamanho batch_size e profundidade 2
        x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 2))
        # y é escalar, pois representa apenas a última previsão da rede
        y = tf.placeholder(tf.float32, shape=())

        prediction = model.predict(x, FLAGS.hidden_layer_size, FLAGS.depth)
        loss = model.loss(prediction, y)
        train_op = model.train(loss, FLAGS.learning_rate)
        evaluation = model.evaluation(prediction, y)

        print(evaluation)

        # cria uma sessão
        session = tf.Session()
        init = tf.global_variables_initializer()

        # merge_all eve ser feito depois da criação do modelo
        summary = tf.summary.merge_all()
        # cria o logger
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

        # para salvar os checkpoints durante o treinamento
        saver = tf.train.Saver()
        # agora que está tudo inicializado...
        session.run(init)

        # treinamento
        step = 0
        elapsed_time_since_last_log = time.time()

        batches = dataset.batch(FLAGS.batch_size)
        batches = batches.repeat()
        batches.shuffle(buffer_size=10000) # randomiza a ordem das batches
        iterator = batches.make_one_shot_iterator()
        next_batch = iterator.get_next()

        for step in range(FLAGS.max_iterations):
            try:
                feed_dict = make_feed_dict(session.run(next_batch), x, y)
            except tf.errors.OutOfRangeError:
                print('Ended first epoch at step ' + str(step))
                print('breaking')
                break

            # roda as operações
            _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)
            
            if step % FLAGS.log_frequency == 0:
                print("Step %d: loss = %.2f (elapsed %.3fs / total %.3fs)" % 
                        (step, loss_value, time.time() - elapsed_time_since_last_log, time.time() - start_time))
                # atualiza o arquivo de eventos
                summary_str = session.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if step % FLAGS.eval_frequency == 0:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(session, checkpoint_file, global_step=step)
                evaluation.evaluate(session, evaluation, x, y, dataset, FLAGS.batch_size)
        
        # salvar o último modelo
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(session, checkpoint_file)

            


def main(_):
    # cria uma pasta dentro de log_dir com a data e hora atual
    path = os.path.normpath(FLAGS.log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d %Hh %Mm %Ss"))
    tf.gfile.MakeDirs(path)
    FLAGS.log_dir = path
    train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50000,
        help="Maximum amount of training iterations"
    )
    parser.add_argument(
        "--hidden-layer-size",
        type=int,
        default=512,
        help="The number of LSTM units per hidden layer"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="The amount of hidden layers the RNN will have"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="The minimum batch size"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="test/data.tfrecord",
        help="The path for getting training data"
    )
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=1000,
        help="How often (in iterations) the training will log the data"
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=1000,
        help="How often (in iterations) the training will log the data"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="test/logs",
        help="The folder to keep logging files"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    