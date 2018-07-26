from __future__ import print_function
import argparse
import os
import sys
from typing import Iterable
import tensorflow as tf

# Modified from https://gist.github.com/sunsided/88d24bf44068fe0fe5b88f09a1bee92a #


parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='The file name of the frozen graph.')
args = parser.parse_args()

if not os.path.exists(args.file):
    parser.exit(1, 'The specified file does not exist: {}'.format(args.file))

graph_def = None
graph = None

print('Loading graph definition ...', file=sys.stderr)
try:
    with tf.gfile.GFile(args.file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
except BaseException as e:
    parser.exit(2, 'Error loading the graph definition: {}'.format(str(e)))

print('Importing graph ...', file=sys.stderr)
try:
    assert graph_def is not None
    with tf.Graph().as_default() as graph:  # type: tf.Graph
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='',
            op_dict=None,
            producer_op_list=None
        )
except BaseException as e:
    parser.exit(2, 'Error importing the graph: {}'.format(str(e)))

print()
print('Operations:')
assert graph is not None
ops = graph.get_operations()  # type: Iterable[tf.Operation]
for op in ops:
    print('- {0:20s} "{1}" ({2} outputs)'.format(op.type, op.name, len(op.outputs)))

print()
print('Sources (operations without inputs):')
for op in ops:
    if len(op.inputs) > 0:
        continue
    print('- {0}'.format(op.name))

print()
print('Operation inputs:')
for op in ops:
    if len(op.inputs) == 0:
        continue
    print('- {0:20}'.format(op.name))
    print('  {0}'.format(', '.join(i.name for i in op.inputs)))

print()
print('Tensors:')
for op in ops:
    for out in op.outputs:
        print('- {0:20} {1:10} "{2}"'.format(str(out.shape), out.dtype.name, out.name))
