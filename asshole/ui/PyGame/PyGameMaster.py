#!/usr/bin/env python
"""
Creates and repeatedly calls an Asshole episode
Holds all the card sprites and player playfields
Processes UI actions
"""
import math

# Only for making the list of Sprites
import pygame
from asshole.core.Episode import Episode, State
from asshole.core.GameMaster import GameMaster
from asshole.ui.PyGame import PyGameCard, PlayerNameLabel
from asshole.ui.PyGame.PyGamePlayer import PyGamePlayer


def player_is_human(player):
    return isinstance(player, PyGamePlayer.PyGamePlayer)

def find_pos(pos, dist, angle):
    """Get a pos, a length and an angle, return the relative position"""
    return pos[0] + math.sin(math.radians(angle)) * dist, pos[1] - math.cos(math.radians(angle)) * dist,


class PyGameMaster(GameMaster):
    def __init__(self, width, height):
        super().__init__()

        # Sprites should be persistent, so init them here
        self.pycards = pygame.sprite.Group()
        for i in range(0, self.deck_size):
            self.pycards.add(PyGameCard(i))

        self.episode = None
        self.current_player = None
        self.active_players = None
        self.positions = []
        self.width = width
        self.height = height
        self.player_status_labels = []
        self.mouse_over = None

    def make_player(self, player_type, name=None):
        super().make_player(player_type, name)
        self.player_status_labels.append(PlayerNameLabel(name))

    def get_pycard(self, c) -> PyGameCard:
        for pc in self.pycards:
            if pc.card.get_index() == c.get_index():
                return pc

    def set_label_pos(self, label, index):
        # TODO: Really should not do this every time
        if index == 0:
            label.rect.x = self.width // 2
            label.rect.y = 2 * self.height // 3 - 120
        elif index == 1:
            label.rect.x = self.width // 3 - 40
            label.rect.y = self.height // 3
        elif index == 2:
            label.rect.x = self.width // 2
            label.rect.y = 120
        elif index == 3:
            label.rect.x = 2 * self.width // 3 + 40
            label.rect.y = self.height // 3

    def play(self, number_of_rounds=100, preset_hands=None):
        # check for any action, and display the current play field
        if not self.episode:
            # Create a new episode
            self.episode = Episode(self.players, self.positions, self.deck, self.listener_list)

        self.positions = self.episode.step()
        if self.episode.state == State.INITIALISED:
            # Do an episode - We need 4 players and a deck of cards.
            pass
        elif self.episode.state == State.DEALING:
            for c in self.pycards:
                c.set_card_params(faceup=False)
        elif self.episode.state == State.SWAPPING:
            pass
        elif self.episode.state == State.ROUND_STARTING:
            pass
        elif self.episode.state == State.PLAYING:
            # Play until the round is won (only one player remaining)
            pass
        elif self.episode.state == State.HAND_WON:
            pass
        elif self.episode.state == State.FINISHED:
            # Set highlight state of all cards to False
            for c in self.pycards:
                c.set_card_params(highlighted=False)
            self.episode = None

        # Find the human current_player (if any)
        human_player_index = 0
        for i, player in enumerate(self.players):
            if player_is_human(player):
                human_player_index = i
                break

        for pc in self.pycards:
            pc.set_card_params(highlighted=False)
        if self.mouse_over:
            for c in self.mouse_over.cards:
                self.get_pycard(c).set_card_params(highlighted=True)

        visible_cards = pygame.sprite.Group()
        visible_others = pygame.sprite.Group()

        # i = 0 is human (or computer), then clockwise
        for i in range(0, 4):
            # Current player is offset by the 'human' index
            # TODO: move human to first place? (locally)
            player_index = (human_player_index + i) % 4
            player = self.players[player_index]

            if self.episode:
                player_meld = self.episode.current_melds[player_index]
            else:
                player_meld = None

            # Render played cards or labels
            if player_meld is None:
                # TODO: Put a label saying "Passed"
                self.player_status_labels[i].set_text(f'{player.name} PASSED')
                self.set_label_pos(self.player_status_labels[i], i)
                visible_others.add(self.player_status_labels[i])
            elif player_meld == '␆':
                # TODO: Put a label saying "Waiting"
                self.player_status_labels[i].set_text(f'{player.name} WAITING')
                self.set_label_pos(self.player_status_labels[i], i)
                visible_others.add(self.player_status_labels[i])
            else:
                # draw the played cards
                for j, card in enumerate(player_meld.cards):
                    pycard = self.pycards.sprites()[card.get_index()]
                    visible_cards.add(pycard)
                    pycard.set_card_params(faceup=True)
                    if i == 0:
                        pos = (self.width // 2 + 40 * j, 2 * self.height // 3 - 120)
                    elif i == 1:
                        pos = (self.width // 3 - 40, self.height // 3 + j * self.height // 12)
                    elif i == 2:
                        pos = (self.width // 2 + 40 * j, 120)
                    elif i == 3:
                        pos = (2 * self.width // 3 + 40, self.height // 3 + j * self.height // 12)
                    pycard.set_card_params(newAngle=i * 90)
                    pycard.rect.x = pos[0]
                    pycard.rect.y = pos[1]

            # Render unplaced cards
            mid_card_index = len(player._hand) / 2.
            for j, card in enumerate(player._hand):
                # index to the pycard
                # It will draw itself, but needs some parameters
                pycard = self.pycards.sprites()[card.get_index()]
                pycard.set_card_params(faceup=i == 0)
                visible_cards.add(pycard)
                card_spread = 6
                card_angle = i * 90 + (j - mid_card_index) * card_spread
                # TODO: more elegant pos and angle to make a nice arc
                pos = (0,0)
                if i == 0:
                    pos = (self.width // 2, self.height)
                elif i == 1:
                    pos = (-50, self.height // 3)
                elif i == 2:
                    pos = (self.width // 2, -pycard.height-20)
                elif i == 3:
                    pos = (self.width, self.height // 3)
                pos = find_pos(pos, 1.5 * pycard.height, card_angle)
                pycard.set_card_params(newAngle = -card_angle)
                # TODO: place by mid-point so hack above can be removed
                pycard.rect.x = pos[0]
                pycard.rect.y = pos[1]

        return visible_cards, visible_others


    def keypress(self, key):
        pass

    def notify_click(self, card):
        """Forward a clicked card to the human player (if any) to decide which to play"""
        # Ignore it if it's not the currently selcted card
        if self.mouse_over and self.mouse_over.cards[-1] == card:
            human = self.get_human_player()
            if human:
                # Let the player decide which to take if multiple clicks
                human.send_card_click(card)

    def get_human_player(self) -> PyGamePlayer:
        for p in self.players:
            if player_is_human(p):
                return p

    def notify_mouseover(self, pycard):
        """Enact mouseover behaviour if """
        if not pycard:
            self.mouse_over = None
        else:
            human = self.get_human_player()
            if not human:
                # No human player
                return
            # Check if it's a valid play, and even then only if it's the best one
            meld = human.get_meld(pycard.card)
            if meld and meld.cards[-1] == pycard.card:
                self.mouse_over = meld

