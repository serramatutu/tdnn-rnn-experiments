import tensorflow as tf
import argparse
import sys

import data
import model
from train import make_feed_dict

FLAGS = None

def evaluate(session, eval_op, x_placeholder, y_placeholder, dataset, batch_size=FLAGS.batch_size):
    print("Evaluating model. Batch size: %d" % (batch_size))
    batches = dataset.batch(batch_size)
    it = batches.make_one_shot_iterator()
    next_batch = it.get_next()

    total_accuracy = 0
    step = 0
    while True:
        try:
            feed_dict = make_feed_dict(session.run(next_batch), x_placeholder, y_placeholder)
            accuracy = session.run(eval_op, feed_dict=feed_dict)
            print("Accuracy at step %d: %0.04f" % (step, accuracy))
            total_accuracy += accuracy

        except tf.errors.OutOfRangeError:
            break
        step += 1
    
    accuracy = total_accuracy / step
    print("Examples: %d | Precision: %0.04f" % (step, accuracy * 100))
    print("Finished evaluation.")

def _main(_):
    sess = tf.Session()
    print('loading meta graph')
    saver = tf.train.import_meta_graph('./final/model.ckpt.meta')
    print('meta graph loaded')
    print('loading checkpoint')
    saver.restore(sess, tf.train.latest_checkpoint('./final'))
    print('checkpoint loaded')

    print('loading dataset')
    dataset = data.QuandlDataset(FLAGS.dataset_path)
    print('dataset loaded')

    print('loading operations')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("Placeholder:0")
    y = graph.get_tensor_by_name("Placeholder_1:0")

    prediction = graph.get_tensor_by_name("rnn_1/strided_slice:0")
    evaluation = model.evaluation(prediction, y)
    print('operations loaded')

    evaluate(sess, evaluation, x, y, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="The minimum batch size"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="test/data.tfrecord",
        help="The path for getting training data"
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=_main, argv=[sys.argv[0]] + unparsed)
