#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gymnasium environment for training an RL agent to play President.

The game runs synchronously on the main thread.  The agent's play()
returns the '␆' sentinel to pause the game loop, giving control back
to the environment so it can return an observation.  On the next
step() call the delivered action is played and the game advances to
the agent's next decision point (or episode end).

Observation: binary vector, size and encoding determined by the model's encode_state
Action:       integer 0–54  (0–53 = meld index, 54 = pass)
Reward:       sparse, at episode end only
              {President: +2, Vice: +1, Citizen: -1, Scumbag: -2}
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from president.core.GameMaster import GameMaster, IllegalPlayPolicy
from president.core.Meld import Meld
from president.core.PlayerRegistry import PlayerEntry, PlayerRegistry
from president.core.AbstractPlayer import AbstractPlayer

ACTION_BITS    = 55   # 0-53 melds, 54 = pass
EXPERIMENTS_DIR = (Path(__file__).resolve().parent.parent / "experiments").resolve()

RANK_REWARDS = {0: 2.0, 1: 1.0, 2: -1.5, 3: -2.5}

PLAY_REWARD     = 0.05   # reward for playing cards (not passing)
HAND_WIN_REWARD = 0.10   # reward for winning a hand

CONFIG_PATH = (Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml").resolve()


def _load_opponents_from_config() -> list[PlayerEntry]:
    """
    Load the 3 opponent entries from config.yaml.
    The RL agent occupies the first seat; the remaining 3 config players
    are used as opponents.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    registry = PlayerRegistry.from_config(cfg)
    entries = registry.all_entries()
    if len(entries) < 3:
        raise ValueError(
            f"config.yaml must define at least 3 non-console players "
            f"(found {len(entries)})"
        )
    return entries[-3:]


class PresidentEnv(gym.Env):
    """
    Single-agent Gymnasium environment for President.
    The RL agent occupies seat 0. Seats 1-3 are filled by opponents
    loaded from config.yaml (or passed in explicitly).
    """

    metadata = {"render_modes": []}

    def __init__(self, encode_fn, obs_size: int,
                 opponents: list[PlayerEntry] | None = None,
                 opponent_pool: list[PlayerEntry] | None = None,
                 debug: bool = False):
        super().__init__()
        self._encode_fn      = encode_fn
        self._opponent_pool  = opponent_pool
        self._opponents      = opponents if opponents is not None else _load_opponents_from_config()
        self._available_pool: list[PlayerEntry] | None = None
        self._reset_count    = 0
        self._debug          = debug
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_BITS)

        self._agent      = None
        self._gm         = None
        self._gm_done    = False
        self._done        = False
        self._final_rank  = None

    # ─────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._done       = False
        self._final_rank = None
        self._agent      = _AgentProxy("Agent", self._encode_fn, obs_size=self.observation_space.shape[0])

        self._gm = GameMaster(policy=IllegalPlayPolicy.PENALISE)
        self._gm.add_player(self._agent)
        for entry in self._sample_opponents():
            self._gm.add_player(entry.player_type(entry.name, **entry.kwargs))
        self._gm.start(number_of_rounds=1)
        self._gm_done = False

        self._advance_game()

        return self._get_observation(), {}

    def step(self, action: int):
        assert not self._done, "Call reset() before step()"

        meld = self._action_to_meld(action)
        self._agent.submit_action(meld)
        self._advance_game()

        if self._done and self._final_rank is None:
            self._final_rank = 3

        obs    = self._get_observation()
        reward = self._get_reward()

        if action != 54:
            reward += PLAY_REWARD
        reward += self._agent.consume_hands_won() * HAND_WIN_REWARD

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
    def _advance_game(self):
        """Step the game forward until the agent needs an action or the episode ends."""
        while not self._gm_done:
            self._gm_done = self._gm.step()
            if self._agent._awaiting_action:
                # early return to get the agent response
                return
        self._final_rank = self._get_agent_rank(self._gm)
        # Game finished normally
        self._done = True
        return

    def _sample_opponents(self) -> list[PlayerEntry]:
        """Return 3 opponents: randomly sampled from pool if one is configured,
        otherwise the fixed opponent list. Pool availability is refreshed every
        500 resets to pick up new .pt files (e.g. splitter_rl_v2 self-play)."""
        if self._opponent_pool is None:
            return self._opponents

        self._reset_count += 1
        if self._available_pool is None or self._reset_count % 500 == 0:
            available = []
            for entry in self._opponent_pool:
                if self._pool_entry_available(entry):
                    available.append(entry)
                else:
                    logging.debug(f"Pool entry '{entry.name}' unavailable — skipping.")
            if len(available) < 3:
                raise RuntimeError(
                    f"Opponent pool has fewer than 3 available entries: "
                    f"{[e.name for e in available]}"
                )
            self._available_pool = available

        return random.sample(self._available_pool, k=3)

    @staticmethod
    def _pool_entry_available(entry: PlayerEntry) -> bool:
        """Check if a pool entry is usable without instantiating it.
        Network players require their .pt weights file to exist."""
        model = entry.kwargs.get('model')
        if model is None:
            return True
        return (EXPERIMENTS_DIR / model / f"{model}.pt").exists()

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


class _AgentProxy(AbstractPlayer):
    """
    Player whose play() cooperates with the synchronous game loop.

    On each decision point play() snapshots state and returns the '␆'
    sentinel, which causes player_turn() to return without advancing.
    The env records the observation and hands control back to the caller.

    On the *next* gm.step() call (after the env delivers an action via
    submit_action), play() is called again and returns the real meld.
    """

    def __init__(self, name: str, encode_fn, obs_size: int):
        super().__init__(name)
        self._encode_fn     = encode_fn
        self._hand_snapshot = []
        self._obs_snapshot  = np.zeros(obs_size, dtype=np.float32)
        self._hands_won     = 0
        self._chosen_meld   = Meld()
        self._awaiting_action = False
        self._has_action      = False

    def play(self) -> Meld:
        """Called by the game engine each time it is this player's turn."""
        if self._has_action:
            # Second call — deliver the action chosen by the env
            self._has_action = False
            self._awaiting_action = False
            return self._chosen_meld

        # First call — snapshot state, then yield control back to the env
        self._hand_snapshot = list(self._hand)
        self._obs_snapshot  = self._encode_fn(self.memory, self).astype(np.float32)
        self._awaiting_action = True
        return '␆'

    def notify_hand_won(self, winner):
        if winner is self:
            self._hands_won += 1
        super().notify_hand_won(winner)

    def consume_hands_won(self) -> int:
        count = self._hands_won
        self._hands_won = 0
        return count

    def submit_action(self, meld: Meld):
        """Called by the environment to deliver the chosen action."""
        self._chosen_meld = meld
        self._has_action = True
