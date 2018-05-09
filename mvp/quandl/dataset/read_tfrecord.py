import tensorflow as tf

# sequence
_ctx_features = {
    "length": tf.FixedLenFeature((), tf.int64)
}
# _seq_features = {
#     "date_differences": tf.VarLenFeature(tf.int64),
#     "value_differences": tf.VarLenFeature(tf.float32)
# }

_seq_features = {
    "date_differences": tf.VarLenFeature(tf.float32),
    "value_differences": tf.VarLenFeature(tf.float32)
}

def _sequence_parser(example):
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

def read_sequence(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(map_func=_sequence_parser)
    return dataset


_features = {
    "date_difference": tf.FixedLenFeature([1], tf.float32),
    "value_difference": tf.FixedLenFeature([1], tf.float32)
}
def _multiple_parser(example):
    a = tf.parse_single_example(
        serialized=example,
        features=_features
    )

    return a["date_difference"], a["value_difference"]

def read_multiple(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(map_func=_multiple_parser)
    return dataset



if __name__ == "__main__":
    import sys
    d = read_multiple(sys.argv[1])
    batches = d.batch(60)
    it = batches.make_one_shot_iterator()
    next_element = it.get_next()
    with tf.Session() as sess:
        count = 0
        while True:
            try:
                sess.run(next_element)
                count += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break

        print(count)