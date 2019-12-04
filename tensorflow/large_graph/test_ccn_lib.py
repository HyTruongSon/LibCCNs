import tensorflow as tf
import numpy as np

import sys
sys.path.append('ccn_lib/')
import ccn1d_grad

# Load the library
ccn1d_lib = tf.load_op_library('ccn_lib/ccn1d_lib.so')
print(dir(ccn1d_lib))

sess = tf.InteractiveSession()