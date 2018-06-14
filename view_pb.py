import tensorflow as tf
from tensorflow.python.platform import gfile

# Modified from https://gist.github.com/jubjamie/2eec49ca1e4f58c5310d72918d991ef6#file-pb_viewer-py

with tf.Session() as sess:
    model_filename ='frozen-graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='logs/tests/1/'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
