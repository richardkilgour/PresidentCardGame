#!/usr/bin/env python
"""
Shared pygame initialisation and layout constants for the PyGame UI.

All other modules in this package import screen, fonts, width/height, and
geometry helpers from here.  Module-level code runs once on first import.
"""
from __future__ import annotations

import math
import os

import pygame
import yaml

from president.players.PyGamePlayer import PyGamePlayer
from president.players.PlayerNaive import PlayerNaive
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSplitter import PlayerSplitter
from president.players.ConsolePlayer import ConsolePlayer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = yaml.safe_load(open(_config_path))

pygame_player_name = next(
    (config[k]['name'] for k in ['player1', 'player2', 'player3', 'player4']
     if config[k].get('type') == 'PyGamePlayer'),
    None,
)

size = width, height = config['screen']['width'], config['screen']['height']

PLAYER_TYPES = {
    'PyGamePlayer':   PyGamePlayer,
    'PlayerNaive':    PlayerNaive,
    'PlayerSimple':   PlayerSimple,
    'PlayerHolder':   PlayerHolder,
    'PlayerSplitter': PlayerSplitter,
    'ConsolePlayer':  ConsolePlayer,
}

# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

STARTUP      = 'startup'
SERVER_INPUT = 'server_input'
GAME_LIST    = 'game_list'
LOBBY        = 'lobby'
OFFLINE      = 'offline'
ONLINE       = 'online'

# ---------------------------------------------------------------------------
# Pygame init
# ---------------------------------------------------------------------------

pygame.init()
screen = pygame.display.set_mode(size)
pygame.display.set_caption('President Card Game')

font_large = pygame.font.SysFont('Arial', 36, bold=True)
font_med   = pygame.font.SysFont('Arial', 22)
font_small = pygame.font.SysFont('Arial', 18)
debug_font = pygame.font.SysFont('Arial', 16)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def find_pos(pos, dist, angle):
    return (pos[0] + math.sin(math.radians(angle)) * dist,
            pos[1] - math.cos(math.radians(angle)) * dist)


# ---------------------------------------------------------------------------
# Text-input widget
# ---------------------------------------------------------------------------

class TextInput(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h, placeholder='', password=False):
        super().__init__()
        self.rect = pygame.Rect(x, y, w, h)
        self.text = ''
        self.placeholder = placeholder
        self.password = password
        self.active = False
        self._font = pygame.font.SysFont('Arial', 20)
        self._render()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            self._render()
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key not in (pygame.K_RETURN, pygame.K_TAB):
                self.text += event.unicode
            self._render()

    def _render(self):
        self.image = pygame.Surface((self.rect.width, self.rect.height))
        self.image.fill(pygame.Color('white'))
        border_col = pygame.Color('royalblue') if self.active else pygame.Color('gray')
        pygame.draw.rect(self.image, border_col,
                         (0, 0, self.rect.width, self.rect.height), 2)
        display = ('•' * len(self.text) if self.password else self.text) or self.placeholder
        col = pygame.Color('black') if self.text else pygame.Color('gray')
        surf = self._font.render(display, True, col)
        self.image.blit(surf, (6, (self.rect.height - surf.get_height()) // 2))
