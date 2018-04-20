import numpy as np
import tensorflow as tf

def _parser(example):
    _ctx_features = {
        "length": tf.FixedLenFeature((), tf.int64)
    }
    _seq_features = {
        "date_differences": tf.VarLenFeature(tf.float32),
        "value_differences": tf.VarLenFeature(tf.float32)
    }
    # converte de example para um tensor esparso não definido
    a, b = tf.parse_single_sequence_example(
        serialized=example,
        context_features=_ctx_features,
        sequence_features=_seq_features
    )

    length = tf.cast(a["length"], tf.int32)

    date_diffs = tf.sparse_tensor_to_dense(b["date_differences"])
    date_diffs = tf.reshape(date_diffs[0], (length - 1,)) # flatten

    value_diffs = tf.sparse_tensor_to_dense(b["value_differences"])
    value_diffs = tf.reshape(value_diffs[0], (length - 1,)) # flatten
    
    return date_diffs, value_diffs

class QuandlSequenceDataset:
    """Dataset que contém uma sequência, que pode ser dividida em batches"""
    
    def __init__(self, sequence):
        print(sequence)
        self._dataset = tf.data.Dataset.from_tensor_slices(sequence)

    def batch(self, batch_size, drop_remainder=True):
        batches = None
        if drop_remainder:
            batches = self._dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            batches = self._dataset.batch(batch_size)

        return batches


class QuandlDataset:
    """Dataset que contém as sequências de dados para treinamento"""
    def _read_from_file(self, filename):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(map_func=_parser)
        self._dataset = dataset

    def __init__(self, filename):
        self._read_from_file(filename)

    def get_initializable_iterator(self):
        """Cria um iterador que percorre todas as sequências"""
        
        return self._dataset.make_initializable_iterator()

    def sequence_batch(self, batch_size, drop_remainder=True):
        """Cria um dataset com batches de sequências"""
        batches = None
        if drop_remainder:
            batches = self._dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            batches = self._dataset.batch(batch_size)
        
        return batches

    def batch(self, batch_size, drop_remainder=True):
        """Cria um dataset com subsequências das sequências contidas"""
        datasets = []

        sequence_iterator = self._dataset.make_one_shot_iterator() # obtém um iterador por todas as sequências
        next_sequence = sequence_iterator.get_next()

        with tf.Session() as sess:
            while True:
                try:
                    # concatena o dataset de retorno com as batches da sequência atual
                    sequence = sess.run(next_sequence)
                    seq_dataset = QuandlSequenceDataset(sequence)
                    datasets.append(seq_dataset.batch(batch_size, drop_remainder))
                except tf.errors.OutOfRangeError:
                    break
        
        return tf.data.Dataset.zip(tuple(datasets))


def main():
    d = QuandlDataset('c:/temp/tensorflow/sequence_data.tfrecord')
    batches = d.batch(10)
    it = batches.make_one_shot_iterator()

    with tf.Session() as sess:
        next_batch = it.get_next()
        while True:
            try:
                print(sess.run(next_batch))

            except tf.errors.OutOfRangeError:
                break

if __name__ == "__main__":
    main()

