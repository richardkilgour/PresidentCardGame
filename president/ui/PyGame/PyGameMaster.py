#!/usr/bin/env python
"""
Creates and repeatedly calls an PresidentCardGame episode
Holds all the card sprites and player playfields
Processes UI actions
"""
import math

# Only for making the list of Sprites
import pygame
from president.core.CardHandler import CardHandler
from president.core.Episode import Episode, State
from president.core.GameMaster import GameMaster
from president.core.IllegalPlayError import IllegalPlayError
from president.ui.PyGame.PyGameCard import PyGameCard
from president.ui.PyGame.GuiElements import PlayerNameLabel
from president.players.PyGamePlayer import PyGamePlayer


def player_is_human(player):
    return isinstance(player, PyGamePlayer)

def find_pos(pos, dist, angle):
    """Get a pos, a length and an angle, return the relative position"""
    return pos[0] + math.sin(math.radians(angle)) * dist, pos[1] - math.cos(math.radians(angle)) * dist,


class PyGameMaster(GameMaster):
    def __init__(self, width, height):
        super().__init__()

        # Sprites should be persistent, so init them here
        self.pycards = pygame.sprite.Group()
        for i in range(0, self.deck.size()):
            self.pycards.add(PyGameCard(i))

        self.current_player = None
        self.active_players = None
        self.width = width
        self.height = height
        self.player_status_labels = []
        self.mouse_over = None

    def make_player(self, player_type, name=None):
        player = player_type(name)
        self.add_player(player)
        self.player_status_labels.append(PlayerNameLabel(name))
        return player

    def get_pycard(self, c) -> PyGameCard:
        for pc in self.pycards:
            if pc.card.same_card(c):
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
        if not self.episode:
            self.reset()

        try:
            self.positions = self.episode.step()
        except IllegalPlayError:
            human = self.get_human_player()
            if human:
                human.next_action = None

        visible_cards = pygame.sprite.Group()
        visible_others = pygame.sprite.Group()

        if self.episode.state == State.FINISHED:
            for c in self.pycards:
                c.set_card_params(highlighted=False)
            self.on_round_completed()
            return visible_cards, visible_others

        if self.episode.state == State.DEALING:
            for c in self.pycards:
                c.set_card_params(faceup=False)

        human_player_index = 0
        for i, player in enumerate(self.player_manager.players):
            if player_is_human(player):
                human_player_index = i
                break

        for pc in self.pycards:
            pc.set_card_params(highlighted=False)
        if self.mouse_over:
            for c in self.mouse_over.cards:
                self.get_pycard(c).set_card_params(highlighted=True)

        for i in range(0, 4):
            player_index = (human_player_index + i) % 4
            player = self.player_manager.players[player_index]
            player_meld = self.episode.current_melds[player_index]

            if player_meld is None:
                self.player_status_labels[i].set_text(f'{player.name} PASSED')
                self.set_label_pos(self.player_status_labels[i], i)
                visible_others.add(self.player_status_labels[i])
            elif player_meld == '␆':
                self.player_status_labels[i].set_text(f'{player.name} WAITING')
                self.set_label_pos(self.player_status_labels[i], i)
                visible_others.add(self.player_status_labels[i])
            else:
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

            mid_card_index = len(player._hand) / 2.
            for j, card in enumerate(player._hand):
                pycard = self.pycards.sprites()[card.get_index()]
                pycard.set_card_params(faceup=i == 0)
                visible_cards.add(pycard)
                card_spread = 6
                card_angle = i * 90 + (j - mid_card_index) * card_spread
                pos = (0, 0)
                if i == 0:
                    pos = (self.width // 2, self.height)
                elif i == 1:
                    pos = (-50, self.height // 3)
                elif i == 2:
                    pos = (self.width // 2, -pycard.height - 20)
                elif i == 3:
                    pos = (self.width, self.height // 3)
                pos = find_pos(pos, 1.5 * pycard.height, card_angle)
                pycard.set_card_params(newAngle=-card_angle)
                pycard.rect.x = pos[0]
                pycard.rect.y = pos[1]

        return visible_cards, visible_others


    def keypress(self, key):
        pass

    def notify_click(self, card):
        """Forward a clicked card to the human player (if any) to decide which to play"""
        # Ignore it if it's not the currently selcted card
        if self.mouse_over and self.mouse_over.cards[-1].same_card(card):
            human = self.get_human_player()
            if human:
                # Let the player decide which to take if multiple clicks
                human.send_card_click(card)

    def get_human_player(self) -> PyGamePlayer:
        for p in self.player_manager.players:
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
            if meld and meld.cards[-1].same_card(pycard.card):
                self.mouse_over = meld

