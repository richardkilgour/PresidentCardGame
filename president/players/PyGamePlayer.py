from president.core.Meld import Meld
from president.players.HumanPlayer import HumanPlayer


class PyGamePlayer(HumanPlayer):
    def __init__(self, name):
        super().__init__(name)
        # next_action: None (waiting), 'PASS', 'QUIT', or a PlayingCard
        self.next_action = None
        self.highlight_meld = None
        self._valid_plays = []

    def notify_player_turn(self, player):
        if player is not self:
            self._valid_plays = []

    def send_keypress(self, key):
        self.next_action = key

    def send_quit(self) -> None:
        """Called by the UI layer (e.g. window close, Escape key)."""
        self.next_action = 'QUIT'

    def on_mouseover(self, card):
        this_meld = self.get_meld(card)
        if this_meld is None:
            pass
        elif not self.highlight_meld:
            self.highlight_meld = this_meld
        elif this_meld.cards[0].get_value() > self.highlight_meld.cards[0].get_value():
            self.highlight_meld = this_meld
        elif (this_meld.cards[0].get_value() == self.highlight_meld.cards[0].get_value()
              and len(this_meld.cards) > len(self.highlight_meld.cards)):
            self.highlight_meld = this_meld
        return self.highlight_meld

    def send_card_click(self, clicked_card):
        """Register a card click; highest-value click wins."""
        if clicked_card == 'PASS':
            self.next_action = 'PASS'
        if self.next_action == 'PASS':
            return
        if self.next_action and self.next_action.get_index() > clicked_card.get_index():
            print(f"Click on {clicked_card} ignored due to {self.next_action}")
        else:
            print(f"Clicked on {clicked_card} replaces {self.next_action}")
            self.next_action = clicked_card

    def show_player(self, i):
        pass

    def get_meld(self, card):
        """Return the valid meld containing this card, or None."""
        for s in self._valid_plays:
            if s.cards and card.same_card(s.cards[-1]):
                return s

    def play(self, valid_plays):
        """
        Return a meld if the UI has registered an action, otherwise '␆' (no-op).
        Raises QuitGame if the player has requested to quit.
        """
        self._valid_plays = valid_plays

        if self.next_action == 'QUIT':
            self.request_quit()

        if self.next_action is None:
            return '␆'

        if self.next_action == 'PASS':
            self.next_action = None
            return Meld()

        m = self.get_meld(self.next_action)
        if m:
            self.next_action = None
            return m

        print(f'INVALID CLICK; {self.next_action} is not a valid play')
        self.next_action = None
        return '␆'
