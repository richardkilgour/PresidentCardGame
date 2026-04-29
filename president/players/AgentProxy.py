import numpy as np

from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld


class AgentProxy(AbstractPlayer):
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
        self._snapshot_plays: list = []

    def play(self, valid_plays) -> Meld:
        """Called by the game engine each time it is this player's turn."""
        if self._has_action:
            # Second call — deliver the action chosen by the env
            self._has_action = False
            self._awaiting_action = False
            assert self._chosen_meld in self._snapshot_plays, (
                f"Agent submitted invalid meld {self._chosen_meld} "
                f"(options: {[s for s in self.valid_plays]}); mask was not respected"
            )
            return self._chosen_meld

        # First call — snapshot state, then yield control back to the env
        self._hand_snapshot  = list(self._hand)
        self._obs_snapshot   = self._encode_fn(self.memory, self).astype(np.float32)
        self._snapshot_plays = valid_plays
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
