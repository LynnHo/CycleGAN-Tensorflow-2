import os, argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

dir = os.path.dirname(os.path.realpath(__file__))

# Modified from https://gist.github.com/moodoki/e37a85fb0258b045c005ca3db9cbc7f6

def freeze_graph(model_folder, output_nodes='a2b_generator/output_image', 
                 output_filename='frozen-graph.pb', 
                 rename_outputs=None):

    #Load checkpoint 
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    output_graph = output_filename

    #Devices should be cleared to allow Tensorflow to control placement of 
    #graph when loading on different machines
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', 
                                       clear_devices=True)

    graph = tf.get_default_graph()

    onames = output_nodes.split(',')

    #https://stackoverflow.com/a/34399966/4190475
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o+':0'), name=n)
            onames=nnames

    input_graph_def = graph.as_graph_def()

    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, input_checkpoint)

        # In production, graph weights no longer need to be updated
        # graph_util provides utility to change all variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, input_graph_def, 
            onames # unrelated nodes will be discarded
        ) 

        # Serialize and write to file
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prune and freeze weights from checkpoints into production models')
    parser.add_argument("--checkpoint_path", 
                        default='./outputs/checkpoints/summer2winter_yosemite',
                        type=str, help="Path to checkpoint files")
    parser.add_argument("--output_nodes", 
                        default='a2b_generator/output_image',
                        type=str, help="Names of output node, comma seperated")
    parser.add_argument("--output_graph", 
                        default='/tmp/frozen-graph.pb',
                        type=str, help="Output graph filename")
    parser.add_argument("--rename_outputs",
                        default=None,
                        type=str, help="Rename output nodes for better \
                        readability in production graph, to be specified in \
                        the same order as output_nodes")
    args = parser.parse_args()

freeze_graph(args.checkpoint_path, args.output_nodes, args.output_graph, args.rename_outputs)
