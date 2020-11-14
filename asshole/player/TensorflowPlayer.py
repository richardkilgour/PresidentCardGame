#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asshole.player.Player import AbstractPlayer
import tensorflow as tf

class TensorFlowPlayer(AbstractPlayer):
    def __init__(self):
        n_input = 240

        # number of units in RNN cell
        n_hidden = 512
        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_input]))
        }

        # reshape to [1, n_input]
        x = tf.reshape(x, [-1, n_input])

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split(x,n_input,1)

        # 1-layer LSTM with n_hidden units.
        rnn_cell = rnn.BasicLSTMCell(n_hidden)

        # generate prediction
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def play(self):
        # Find the possible melded, and play the lowest one, unless we have a double
        minimum_value = self.game_master.current_meld()
        selection = self.possible_plays(minimum_value)
        return selection[0]

    def notify_hand_start(self, starter):
        # TODO: Clear all the inputs
        pass

    def notify_hand_won(self, winner):

        pass

    def notify_played_out(self, opponent, pos):
        pass

    def notify_play(self, player, meld):
        # Add what was played to the set of inputs
        pass
