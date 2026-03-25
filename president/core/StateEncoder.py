#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StateEncoder converts the current game state into a fixed-size binary vector
for use in supervised and reinforcement learning.

State vector layout (341 bits total):

  Block 0 — current player now          57 bits
    state:      3 bits  [waiting, has_played, won_round]
    own hand:  54 bits  (cumulative count per value)

  Blocks 1-4 — last 4 history events   71 bits each
    state:      3 bits  [waiting, has_played, won_round]
    hand size: 14 bits  (unary, 0-13)
    last meld: 54 bits  (one-hot)

Implicit from encoding:
  passed   → zero meld + not waiting + not won_round
  finished → zero hand size

Action / meld vector (54 bits, one-hot):
  [3x1, 3x2, 3x3, 3x4, 4x1, ... 2x4, Jx1, Jx2]
  pass → all zeros

All state is derived from PlayHistory and GameEvent.hand.
No live player state is used.
"""
from __future__ import annotations

import numpy as np
from collections import Counter

from president.core.Meld import Meld
from president.core.PlayHistory import EventType, GameEvent


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

STATE_BITS     = 3    # [waiting, has_played, won_round]
HAND_SIZE_BITS = 14   # unary, 0-13 cards
MELD_BITS      = 54   # one-hot over all possible melds
HAND_BITS      = 54   # cumulative count per value

BLOCK_0_SIZE   = STATE_BITS + HAND_BITS                    # 57
BLOCK_N_SIZE   = STATE_BITS + HAND_SIZE_BITS + MELD_BITS   # 71
TOTAL_BITS     = BLOCK_0_SIZE + BLOCK_N_SIZE * 4           # 341

# State bit indices
IDX_WAITING    = 0
IDX_HAS_PLAYED = 1
IDX_WON_ROUND  = 2

# Meld layout: value * 4 + (count - 1), jokers at offset 52
MAX_REGULAR_VALUE = 13
JOKER_OFFSET      = 52  # 13 * 4


class StateEncoder:
    """
    Encodes game state into a fixed-size binary numpy vector.

    Two entry points:
      encode()            — live encoding during play, reads player.memory
      encode_from_event() — history replay, reads from GameEvent and
                            history slice only. No live player state used.

    Usage:
        encoder = StateEncoder()

        # During play (called from player.play())
        vector = encoder.encode(player)

        # During trajectory building (called from EpisodeEncoder)
        vector = encoder.encode_from_event(event, history_slice)

        # Encode/decode actions
        action = StateEncoder.encode_meld(meld)
        meld   = StateEncoder.decode_meld(action_vector)
    """

    # ─────────────────────────────────────────────
    # Public entry points
    # ─────────────────────────────────────────────

    def encode(self, current_player) -> np.ndarray:
        """
        Encode the current game state for a live player.
        Reads from current_player.memory and current_player._hand.

        Args:
            current_player: The player whose turn it is.

        Returns:
            Binary numpy array of shape (341,)

        Raises:
            RuntimeError: If the player's memory is empty.
        """
        history = current_player.memory

        if not history._memory:
            raise RuntimeError(
                f"StateEncoder called with empty memory for "
                f"{current_player.name}. notify_hand_start and "
                f"notify_player_turn should have fired first."
            )

        # Find the most recent MELD event for this player to use as
        # block 0 — but use current_player._hand for the live hand
        current_event = history.last_event_for(current_player)

        return self._encode_blocks(
            state_bits=self._state_from_event(current_event),
            hand=current_player._hand,
            history_slice=history._memory,
        )

    def encode_from_event(self, event: GameEvent,
                          history_slice: list) -> np.ndarray:
        """
        Encode state at a past decision point using history alone.
        Uses event.hand for block 0 — no live player state needed.

        Args:
            event:         The MELD GameEvent at the decision point.
                           Must have event.hand populated.
            history_slice: history._memory[:event_index + 1] —
                           history up to and including this event.

        Returns:
            Binary numpy array of shape (341,)
        """
        return self._encode_blocks(
            state_bits=self._state_from_event(event),
            hand=event.hand or [],
            history_slice=history_slice,
        )

    # ─────────────────────────────────────────────
    # Core encoder
    # ─────────────────────────────────────────────

    def _encode_blocks(self, state_bits: np.ndarray,
                       hand: list,
                       history_slice: list) -> np.ndarray:
        """
        Build the full 341-bit state vector from components.

        Args:
            state_bits:    3-bit state for block 0.
            hand:          Card list for block 0 hand encoding.
            history_slice: Event list for blocks 1-4.
        """
        vec = np.zeros(TOTAL_BITS, dtype=np.int8)
        offset = 0

        # --- Block 0: current player state + hand ---
        vec[offset:offset + STATE_BITS] = state_bits
        offset += STATE_BITS

        vec[offset:offset + HAND_BITS] = self._encode_hand(hand)
        offset += HAND_BITS

        # --- Blocks 1-4: last 4 events, most recent first ---
        last_events = history_slice[-4:]
        for e in reversed(last_events):
            offset = self._encode_historic_block(vec, offset, e)
        # Pad with zero blocks if fewer than 4 events in history
        for _ in range(4 - len(last_events)):
            offset += BLOCK_N_SIZE

        assert offset == TOTAL_BITS, \
            f"State vector length mismatch: {offset} != {TOTAL_BITS}"
        return vec

    # ─────────────────────────────────────────────
    # Block encoders
    # ─────────────────────────────────────────────

    def _encode_historic_block(self, vec: np.ndarray, offset: int,
                                event: GameEvent) -> int:
        """
        Encode a historic event block (blocks 1-4).
        Layout: state (3) + hand_size (14) + meld (54) = 71 bits
        """
        vec[offset:offset + STATE_BITS] = self._state_from_event(event)
        offset += STATE_BITS

        vec[offset:offset + HAND_SIZE_BITS] = self._encode_hand_size(
            event.remaining_cards
        )
        offset += HAND_SIZE_BITS

        meld = event.meld if isinstance(event.meld, Meld) else Meld()
        vec[offset:offset + MELD_BITS] = self.encode_meld(meld)
        offset += MELD_BITS

        return offset

    # ─────────────────────────────────────────────
    # State helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _state_from_event(event: GameEvent | None) -> np.ndarray:
        """
        Derive 3-bit state vector from a GameEvent.
        None → waiting (no history yet).

        passed   → all zeros, inferred from zero meld
        finished → all zeros, inferred from zero hand size
        """
        state = np.zeros(STATE_BITS, dtype=np.int8)
        if event is None:
            state[IDX_WAITING] = 1
        elif event.event_type == EventType.ROUND_WON:
            state[IDX_WON_ROUND] = 1
        elif event.event_type == EventType.MELD and event.meld.cards:
            state[IDX_HAS_PLAYED] = 1
        return state

    # ─────────────────────────────────────────────
    # Encoding primitives
    # ─────────────────────────────────────────────

    @staticmethod
    def encode_meld(meld: Meld) -> np.ndarray:
        """
        Encode a meld as a one-hot 54-bit vector.
        Pass (empty meld) → all zeros.

        Bit position = value * 4 + (count - 1)
        Jokers: 52 + (count - 1)
        """
        vec = np.zeros(MELD_BITS, dtype=np.int8)
        if not meld or not meld.cards:
            return vec
        value = meld.cards[0].get_value()
        count = len(meld)
        pos = JOKER_OFFSET + (count - 1) if value == MAX_REGULAR_VALUE \
            else value * 4 + (count - 1)
        vec[pos] = 1
        return vec

    @staticmethod
    def decode_meld(vec: np.ndarray) -> Meld:
        """
        Decode a one-hot 54-bit vector back to a Meld.
        All zeros → pass (empty Meld).
        """
        from president.core.PlayingCard import PlayingCard
        indices = np.where(vec == 1)[0]
        if len(indices) == 0:
            return Meld()
        pos = int(indices[0])
        if pos >= JOKER_OFFSET:
            value = MAX_REGULAR_VALUE
            count = pos - JOKER_OFFSET + 1
        else:
            value = pos // 4
            count = pos % 4 + 1
        meld = None
        for _ in range(count):
            meld = Meld(PlayingCard(value * 4), meld)
        return meld

    @staticmethod
    def _encode_hand(hand: list) -> np.ndarray:
        """
        Encode a hand as a 54-bit cumulative count vector.
        For each value, bits 1..count are set, rest are zero.

        e.g. 3 cards of value 3, 1 card of value 4:
             [1,1,1,0, 1,0,0,0, 0,0,0,0, ...]
        """
        vec = np.zeros(HAND_BITS, dtype=np.int8)
        counts = Counter(c.get_value() for c in hand)
        for value, count in counts.items():
            offset = JOKER_OFFSET if value == MAX_REGULAR_VALUE \
                else value * 4
            max_count = 2 if value == MAX_REGULAR_VALUE else 4
            for i in range(min(count, max_count)):
                vec[offset + i] = 1
        return vec

    @staticmethod
    def _encode_hand_size(num_cards: int) -> np.ndarray:
        """
        Encode hand size as a unary 14-bit vector.
        e.g. 5 cards → [1,1,1,1,1,0,0,0,0,0,0,0,0,0]
        """
        vec = np.zeros(HAND_SIZE_BITS, dtype=np.int8)
        vec[:min(num_cards, HAND_SIZE_BITS)] = 1
        return vec


# ─────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────

def _test_meld_encoding():
    from president.core.PlayingCard import PlayingCard

    assert np.sum(StateEncoder.encode_meld(Meld())) == 0

    single_3 = Meld(PlayingCard(0))
    vec = StateEncoder.encode_meld(single_3)
    assert vec[0] == 1 and np.sum(vec) == 1

    double_3 = Meld(PlayingCard(1), single_3)
    vec = StateEncoder.encode_meld(double_3)
    assert vec[1] == 1 and np.sum(vec) == 1

    joker = Meld(PlayingCard(52))
    vec = StateEncoder.encode_meld(joker)
    assert vec[52] == 1 and np.sum(vec) == 1

    double_joker = Meld(PlayingCard(53), joker)
    vec = StateEncoder.encode_meld(double_joker)
    assert vec[53] == 1 and np.sum(vec) == 1

    for meld in [single_3, double_3, joker, double_joker, Meld()]:
        decoded = StateEncoder.decode_meld(StateEncoder.encode_meld(meld))
        assert len(decoded) == len(meld), \
            f"Round trip failed: {meld} → {decoded}"

    print("Meld encoding tests passed.")


def _test_hand_encoding():
    from president.core.PlayingCard import PlayingCard

    hand = [PlayingCard(0), PlayingCard(1), PlayingCard(2), PlayingCard(4)]
    vec = StateEncoder._encode_hand(hand)
    assert vec[0] == 1 and vec[1] == 1 and vec[2] == 1 and vec[3] == 0
    assert vec[4] == 1 and vec[5] == 0
    print("Hand encoding tests passed.")


def _test_hand_size_encoding():
    vec = StateEncoder._encode_hand_size(5)
    assert list(vec) == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert np.sum(StateEncoder._encode_hand_size(0)) == 0
    assert np.sum(StateEncoder._encode_hand_size(13)) == 13
    print("Hand size encoding tests passed.")


if __name__ == '__main__':
    _test_meld_encoding()
    _test_hand_encoding()
    _test_hand_size_encoding()
    print("All StateEncoder tests passed.")
