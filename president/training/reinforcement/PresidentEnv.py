#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium environment for training an RL agent to play President.

The game runs on a background thread. The agent communicates with it
via threading.Event handshakes — no busy-waiting.

Observation: binary vector, size determined by model_cls.INPUT_SIZE
Action:       integer 0–54  (0–53 = meld index, 54 = pass)
Reward:       sparse, at episode end only
              {President: +2, Vice: +1, Citizen: -1, Scumbag: -2}
"""
from __future__ import annotations

import logging
import threading
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from president.core.GameMaster import GameMaster, IllegalPlayPolicy
from president.core.Meld import Meld
from president.core.PlayerRegistry import PlayerRegistry
from president.models.single_meld_mlp import SingleMeldMLP
from president.players.PlayerSplitter import PlayerSplitter

ACTION_BITS = 55   # 0-53 melds, 54 = pass

RANK_REWARDS = {0: 2.0, 1: 1.0, 2: -1.0, 3: -2.0}


class PresidentEnv(gym.Env):
    """
    Single-agent Gymnasium environment for President.
    The agent occupies seat 0. Seats 1-3 are fixed opponents.
    """

    metadata = {"render_modes": []}

    def __init__(self, model_cls=SingleMeldMLP, opponent_type=PlayerSplitter):
        super().__init__()
        self._model_cls        = model_cls
        self.opponent_type     = opponent_type
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(model_cls.INPUT_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_BITS)

        self._registry = PlayerRegistry()
        self._registry.register(opponent_type, name="Opponent")

        self._agent       = None
        self._game_thread = None
        self._done        = False
        self._final_rank  = None

    # ─────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._game_thread and self._game_thread.is_alive():
            self._agent.abort()
            self._game_thread.join()

        self._done       = False
        self._final_rank = None
        self._agent      = _AgentProxy("Agent", self._model_cls)

        gm = GameMaster(
            registry=self._registry,
            policy=IllegalPlayPolicy.PENALISE,
        )
        gm.add_player(self._agent)
        for i in range(1, 4):
            gm.make_player("Opponent", f"Opponent_{i}")
        gm.start(number_of_rounds=1)

        self._game_thread = threading.Thread(
            target=self._run_game, args=(gm,), daemon=True
        )
        self._game_thread.start()
        self._agent.wait_for_turn()

        return self._get_observation(), {}

    def step(self, action: int):
        assert not self._done, "Call reset() before step()"

        meld = self._action_to_meld(action)
        self._agent.submit_action(meld)
        self._agent.wait_for_turn()

        if self._done and self._final_rank is None:
            self._final_rank = 3

        obs        = self._get_observation()
        reward     = self._get_reward()
        terminated = self._done
        truncated  = False
        info       = {"final_rank": self._final_rank} if self._done else {}

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(ACTION_BITS, dtype=bool)
        legal = self._agent.possible_plays()
        for meld in legal:
            mask[self._meld_to_action(meld)] = True
        return mask

    def _get_observation(self) -> np.ndarray:
        return self._agent._obs_snapshot

    def _get_reward(self) -> float:
        if not self._done:
            return 0.0
        return RANK_REWARDS.get(self._final_rank, 0.0)

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _run_game(self, gm: GameMaster):
        try:
            done = False
            while not done:
                done = gm.step()
            self._final_rank = self._get_agent_rank(gm)
        except Exception as e:
            logging.error(f"Game thread error: {e}", exc_info=True)
            self._final_rank = 3
        finally:
            self._done = True
            self._agent.signal_done()

    def _get_agent_rank(self, gm: GameMaster) -> int | None:
        ranks = gm.episode.ranks
        if self._agent in ranks:
            return ranks.index(self._agent)
        return None

    def _action_to_meld(self, action: int) -> Meld:
        if action == 54:
            return Meld()
        value = action // 4
        count = action % 4 + 1
        # Use snapshot hand — consistent with the mask that was computed
        candidates = [c for c in self._agent._hand_snapshot
                      if c.get_value() == value]
        if len(candidates) < count:
            return Meld()
        meld = None
        for card in candidates[:count]:
            meld = Meld(card, meld) if meld else Meld(card)
        return meld

    @staticmethod
    def _meld_to_action(meld: Meld) -> int:
        if not meld.cards:
            return 54
        value = meld.cards[0].get_value()
        count = len(meld.cards)
        return value * 4 + count - 1


class _AgentProxy(PlayerSplitter):
    """
    Player that blocks the game thread at each decision point
    until the environment delivers an action via submit_action().
    State is snapshotted at decision time so the main thread always
    sees a consistent view regardless of subsequent game thread updates.
    """

    def __init__(self, name: str, model_cls):
        super().__init__(name)
        self._model_cls     = model_cls
        self._action_needed = threading.Event()
        self._action_ready  = threading.Event()
        self._chosen_meld   = Meld()
        self._aborted       = False
        self._done_flag     = False
        self._state_lock    = threading.Lock()
        self._hand_snapshot = []
        self._obs_snapshot  = np.zeros(model_cls.INPUT_SIZE, dtype=np.float32)

    def play(self) -> Meld:
        """Called by the game thread — snapshot state then block until action arrives."""
        with self._state_lock:
            self._hand_snapshot = list(self._hand)
            self._obs_snapshot  = self._model_cls.encode_state(self.memory, self).astype(np.float32)

        self._action_needed.set()
        if not self._action_ready.wait(timeout=30):
            raise RuntimeError(
                "Deadlock: play() timed out waiting for action after 30s"
            )
        self._action_ready.clear()
        if self._aborted:
            return Meld()
        return self._chosen_meld

    def notify_play(self, player, meld):
        with self._state_lock:
            super().notify_play(player, meld)

    def notify_hand_won(self, winner):
        with self._state_lock:
            super().notify_hand_won(winner)

    def submit_action(self, meld: Meld):
        """Called by the environment thread to deliver an action."""
        self._chosen_meld = meld
        self._action_ready.set()

    def wait_for_turn(self):
        """Called by the environment — polls until play() is called or episode ends."""
        while not self._action_needed.wait(timeout=1.0):
            if self._aborted or self._done_flag:
                return
        self._action_needed.clear()

    def signal_done(self):
        """Called by the game thread when the episode ends."""
        self._done_flag = True
        self._action_needed.set()

    def abort(self):
        """Unblock both events so threads can exit cleanly."""
        self._aborted   = True
        self._done_flag = True
        self._action_ready.set()
        self._action_needed.set()
