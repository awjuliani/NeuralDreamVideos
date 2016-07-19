import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.input_size])
        self.targets = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.input_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        self.list_data = tf.split(1,args.seq_length,self.input_data)
        self.list_data = [tf.squeeze(input_, [1]) for input_ in self.list_data]
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.input_size])
            softmax_b = tf.get_variable("softmax_b", [args.input_size])

        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return prev_symbol

        def ssLoss(logit,target):
            return tf.square(target - logit)

        def MYsequence_loss_by_example(logits, targets, weights,
                                     average_across_timesteps=True,
                                     softmax_loss_function=None, name=None):
          if len(targets) != len(logits) or len(weights) != len(logits):
            raise ValueError("Lengths of logits, weights, and targets must be the same "
                             "%d, %d, %d." % (len(logits), len(weights), len(targets)))
          with ops.op_scope(logits + targets + weights, name,
                            "sequence_loss_by_example"):
            log_perp_list = []
            for logit, target, weight in zip(logits, targets, weights):
              if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    logit, target)
              else:
                crossent = softmax_loss_function(logit, target)
              print crossent, weight
              log_perp_list.append(crossent * weight)
              print log_perp_list              
            log_perps = math_ops.add_n(log_perp_list)
            if average_across_timesteps:
              total_size = math_ops.add_n(weights)
              total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
              log_perps /= total_size
          return log_perps

        self.outputs, last_state = tf.nn.seq2seq.rnn_decoder(self.list_data, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        self.output = tf.reshape(tf.concat(1, self.outputs), [-1, args.rnn_size])
        self.target = tf.reshape(tf.concat(1, self.targets), [-1, args.input_size])
        self.logits = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b)
        self.loss = MYsequence_loss_by_example([self.logits],[self.target],[tf.ones([args.batch_size * args.seq_length,args.input_size],tf.float32)],softmax_loss_function=ssLoss)
        self.cost = tf.reduce_sum(self.loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))