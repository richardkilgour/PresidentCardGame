from president.core.AbstractPlayer import AbstractPlayer
from president.core.Meld import Meld
from president.players.PlayerSimple import PlayerSimple


class PyGamePlayer(PlayerSimple):
    def __init__(self, name):
        AbstractPlayer.__init__(self, name)
        # NextAction can be None (no action) or a list or cards to play (empty list = pass)
        self.next_action = None
        self.highlight_meld = None
        self._valid_plays = []

    def notify_player_turn(self, player):
        if player is not self:
            self._valid_plays = []

    def send_keypress(self, key):
        self.next_action = key

    def on_mouseover(self, card):
        # Like send_card_click, need to see if this card is larger then the other mouseover cards
        # Return the best meld to date
        this_meld = self.get_meld(card)
        if this_meld == None:
            # Not a valid card
            pass
        # Higher index = forget the previous best
        elif not self.highlight_meld:
            self.highlight_meld = this_meld
        elif this_meld.cards[0].get_value() > self.highlight_meld.cards[0].get_value():
            self.highlight_meld = this_meld
        elif this_meld.cards[0].get_value() == self.highlight_meld.cards[0].get_value() and \
                len(this_meld.cards) > len(self.highlight_meld.cards):
            self.highlight_meld = this_meld
        return self.highlight_meld


    def send_card_click(self, clicked_card):
        """Find the highest value card that was clicked (in case many cards are clicked)"""
        # If this is a card in the hand, play it
        if clicked_card == 'PASS':
            self.next_action = 'PASS'
        if self.next_action == 'PASS':
            return
        if self.next_action and self.next_action.get_index() > clicked_card.get_index():
            # Only if it's higher than all the existing 'action' cards, replace them
            print(f"Click on {clicked_card} ignored due to {self.next_action}")
        else:
            print(f"Clicked on {clicked_card} replaces {self.next_action}")
            self.next_action = clicked_card

    def show_player(self, i):
        pass

    def get_meld(self, card):
        """Return a meld if the card can be turned into a valid meld, otherwise None"""
        for s in self._valid_plays:
            if s.cards and card.same_card(s.cards[-1]):
                return s

    def play(self, valid_plays):
        """
        Process an action set by another thread
        Return a meld (set of cards) if a valid card was clicked
        Return a pass (empty Meld) if the clicked card is an empty set
        Fall through if no action is selected yet (return noop '␆')
        """
        self._valid_plays = valid_plays
        if self.next_action is None:
            return '␆'
        if self.next_action == 'PASS':
            self.next_action = None
            return Meld()

        m = self.get_meld(self.next_action)
        # Logic for multiple selections relies on highest card not being on lower combos
        if m:
            self.next_action = None
            return m
        print(f'INVALID CLICK; {self.next_action} is not a valid play')
        self.next_action = None
        return '␆'
