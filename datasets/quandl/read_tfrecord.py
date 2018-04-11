import tensorflow as tf

_ctx_features = {
    "length": tf.FixedLenFeature((), tf.int64)
}
_seq_features = {
    "date_differences": tf.VarLenFeature(tf.int64),
    "value_differences": tf.VarLenFeature(tf.float32)
}

def _parser(example):
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

def read(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(map_func=_parser)
    return dataset
