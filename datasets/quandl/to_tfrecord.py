# gera dados baseado na variação

import json
import tensorflow as tf
from datetime import datetime

with open("data.json") as json_data:
    data = json.load(json_data)

# reverte a lista para ficar do passado ao presente
ordered_data = data["dataset_data"]["data"][::-1]

last = ordered_data[0]
last[0] = datetime.strptime(last[0], "%Y-%m-%d")

#listas para serem colocadas no exemplo
date_diffs = []
value_diffs = []

for x in ordered_data[1:]: # percorre do atual para o passado
    #lê a data
    date = datetime.strptime(x[0], "%Y-%m-%d") # lê a data do acontecimento

    # coloca os dados no exemplo
    date_diffs.append((date - last[0]).days)
    value_diffs.append(x[1] - last[1])

    #atualiza o último
    last[0] = date
    last[1] = x[1]

# feature = tf.train.Feature(
#             value = {
#                 "date_difference": tf.train.Feature(int64_list=tf.train.Int64List(value=date_diffs)),
#                 "value_difference": tf.train.Feature(float_list=tf.train.FloatList(value=value_diffs))
#             }
#         )

# cria o example

ex = tf.train.SequenceExample(
    context = {
        "feature": {
            "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(ordered_data)]))
        }
    },
    feature_lists = {
        "feature_list": {
            "date_differences": tf.train.FeatureList(feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=date_diffs))]),
            "value_differences": tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=value_diffs))])
        }
    }
)
print(ex)

writer = tf.python_io.TFRecordWriter("data.tfrecord")
writer.write(ex.SerializeToString())
writer.close()
