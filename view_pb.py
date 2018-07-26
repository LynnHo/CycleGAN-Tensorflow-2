import tensorflow as tf
import argparse
from tensorflow.python.platform import gfile

""" param """
parser = argparse.ArgumentParser(description='')
parser.add_argument('--graph', dest='graph_filename', default='graph.pb', help='which graph to use')
args = parser.parse_args()

graph_filename = args.model_filename

# Modified from https://gist.github.com/jubjamie/2eec49ca1e4f58c5310d72918d991ef6#file-pb_viewer-py

with tf.Session() as sess:
    with gfile.FastGFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='logs/' + graph_filename
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
