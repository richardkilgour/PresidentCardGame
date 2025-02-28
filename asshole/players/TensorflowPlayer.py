#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from collections import deque

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from asshole.core.AbstractPlayer import AbstractPlayer
from asshole.core.Meld import Meld

MAX_STATES = 15000


class State:
    def __init__(self, positions):
        self.positions = positions
        # Cards before the swaps
        self.initial_deal = []
        # Cards after the swap
        self.cards = []
        self.melds = []
        self.next_player = 0

    def get_input_vector(self):
        """
        Use the current State to create an input vector for the network
        First 16 bits are the positions for this round (if any)
        Then 54 bits for the initial hand (before card swaps)
        Then 54 bits for the hand after card swaps
        Then 56 bits for every action taken in the game, starting with self
        then going clockwise through the opponents
        If players can't play, use NoOp (Index 55)
        Otherwise it's a meld or a pass (Index 54)
        """
        vector = [0] * 16
        if self.positions:
            for i, p in enumerate(self.positions):
                vector[i*4+p] = 1
        vector.extend(self.initial_deal)
        vector.extend(self.cards)
        for m in self.melds:
            vector.extend(m)
        # Buffer
        vector += [0] * (MAX_STATES - len(vector))
        return vector

    @staticmethod
    def meld_to_vector(meld):
        """
        Return a vector of 56 bits (Well, ints, but all 0 or 1)
        Meld of None is a NoOp (Index 55, but should be be trained on)
        Meld without cards is a Pass (Index 54)
        Otherwise it's the index of the meld's card(s)
        """
        vector = [0] * 56
        if not meld.cards:
            vector[54] = 1
        else:
            for c in meld.cards:
                index = 4 * c.get_value() + c.get_suit()
                vector[index] = 1
        return vector

    def register_meld(self, index, meld):
        while index != self.next_player:
            # Add a 'Skip' vector
            self.melds.append([0] * 55 + [1])
            self.next_player += 1
            self.next_player %= 4
        self.melds.append(self.meld_to_vector(meld))


class TensorflowPlayer(AbstractPlayer):
    def __init__(self, name):
        AbstractPlayer.__init__(self, name)
        # Network specific things
        self._action_size = 55
        self._state_size = MAX_STATES
        self._optimizer = Adam()
        self._epsilon = 0.05
        self.q_network = self.create_network()
        self.target_network = self.create_network()
        self.experience_replay = deque(maxlen=2000)

        # Game specific things for the listener to pass to the State
        self.positions = None
        self.state = State(self.positions)
        self.old_state = None
        self.action = None
        self.ready_to_train = False

    def play(self):
        # Before playing, remember the state for training purposes
        if self.old_state:
            next_state = self.state.get_input_vector()
            action = State.meld_to_vector(self.action)
            terminated = False
            self.experience_replay.append((self.old_state, action, 0, next_state, terminated))
            self.old_state = next_state
            if len(self.experience_replay) == 100:
                self.ready_to_train = True
        else:
            # No actions taken yet, so this is the initial state
            assert(self.action is None)
            self.old_state = self.state.get_input_vector()
        # Use epsilon to decide on the best action or a random one
        if np.random.rand() <= self._epsilon:
            possible_plays = self.possible_plays(self.target_meld)
            self.action = random.choice(possible_plays)
        else:
            x = np.reshape(np.array(self.state.get_input_vector(), dtype=np.bool), (1,-1))
            q_values = self.q_network.predict(x)
            # Should use a mode-free thing here, but just follow the rules
            q_values = q_values[0] # Batch size = 1

            while True:
                best_play_index = np.argmax(q_values)
                meld = self.index_to_meld(best_play_index)
                if meld and (not meld.cards or not self.target_meld or meld > self.target_meld):
                    break
                else:
                    q_values[best_play_index] = -1
            self.action = meld
        return self.action

    def notify_hand_won(self, winner):
        super(TensorflowPlayer, self).notify_hand_won(winner)

    def notify_played_out(self, opponent, pos):
        super(TensorflowPlayer, self).notify_played_out(opponent, pos)
        # If I've played out, then I can get a reward based on my position
        # Fairly arbitrary, actually, as long as it's inversely proportional to pos
        if opponent == self:
            new_reward = (1.5 - pos) * 100.
            (state, action, reward, next_state, terminated) = self.experience_replay[-1]
            self.experience_replay[-1] = (state, action, new_reward, next_state, True)
            if self.ready_to_train:
                self.retrain(len(self.experience_replay))
                self.experience_replay = deque(maxlen=2000)
                self.ready_to_train = False

    def notify_play(self, player, meld):
        super(TensorflowPlayer, self).notify_play(player, meld)
        # Add what was played to the set of inputs
        if player == self:
            self.state.register_meld(0, meld)
        else:
            self.state.register_meld(self.players.get_index(player) + 1, meld)

    def notify_hand_start(self, starter):
        super(TensorflowPlayer, self).notify_hand_start(starter)
        self.state = State(self.positions)

    def index_to_meld(self, index):
        """
        Find the card value, and the number of cards
        Actually finds the number of cards of the given value, NOT sensitive to suit
        """
        value = index // 4
        count = index % 4
        meld = Meld()
        # Pass?
        if index == 54:
            return meld
        for c in self._hand:
            if c.get_value() == value:
                meld = Meld(c, meld)
                if count == 0:
                    return meld
                count -= 1
        # The index was not a valid set of cards
        return None

    def create_network(self):
        # Create a network
        model = Sequential()
        model.add(Input(shape=(self._state_size,), dtype=tf.bool))
        # Use an embedding layer? Maybe a few (go to functional model)
        #model.add(Embedding(self._state_size, 10, input_length=1))
        #model.add(Reshape((10,)))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self._action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def retrain(self, batch_size):
        minibatch = random.sample(self.experience_replay, batch_size)
        for state, action, reward, next_state, terminated in minibatch:
            state = np.array(state, dtype=np.bool)
            print(state.shape)
            target = self.q_network.predict(state)
            if terminated:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=1, verbose=0)