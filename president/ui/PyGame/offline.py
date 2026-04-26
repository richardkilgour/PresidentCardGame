#!/usr/bin/env python
"""
Offline (local engine) mode for the PyGame UI.

init()   — create and configure a PyGameMaster from config
handle() — process pygame events
draw()   — render the current game frame
"""
from __future__ import annotations

import pygame
from pygame.locals import KEYDOWN

from president.core.AbstractPlayer import AbstractPlayer
from president.players.PyGamePlayer import PyGamePlayer
from president.ui.PyGame.GuiElements import PlayerNameLabel, PassButton, StatBox
from president.ui.PyGame.PyGameMaster import PyGameMaster
from president.ui.PyGame.app import (
    screen, width, height, debug_font,
    config, PLAYER_TYPES,
)

# Seat-to-screen-position mapping (matches PyGameMaster.set_label_pos / online rendering)
NAME_LABEL_POSITIONS = [
    (width // 2,    height - 60),   # seat 0 — bottom
    (0,             height // 3),   # seat 1 — left
    (width // 2,    0),             # seat 2 — top
    (width - 60,    height // 3),   # seat 3 — right
]

_gm:          PyGameMaster | None = None
_card_sprites = pygame.sprite.Group()
_ui_sprites   = pygame.sprite.Group()
_human:       PyGamePlayer | None = None
_pass_btn:    PassButton | None   = None
_stat_box:    StatBox | None      = None


def init():
    global _gm, _card_sprites, _ui_sprites, _human, _pass_btn, _stat_box

    _gm = PyGameMaster(width, height)
    _card_sprites = pygame.sprite.Group()
    _ui_sprites   = pygame.sprite.Group()

    players = [config['player1'], config['player2'], config['player3'], config['player4']]
    # Rotate so the PyGamePlayer sits at index 0 (bottom of screen)
    for _ in range(len(players)):
        if players[0]['type'] == 'PyGamePlayer':
            break
        players.append(players.pop(0))

    for i, p in enumerate(players):
        _gm.make_player(PLAYER_TYPES[p['type']], p['name'])
        nx, ny = NAME_LABEL_POSITIONS[i]
        _ui_sprites.add(PlayerNameLabel(p['name'], nx, ny))

    _human    = _gm.get_human_player()
    _pass_btn = PassButton(width // 2, height // 2)
    _ui_sprites.add(_pass_btn)
    _stat_box = StatBox(0, height - 150)
    _ui_sprites.add(_stat_box)


def handle(events, mouse_pos):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if _pass_btn and _pass_btn.rect.collidepoint(event.pos) and _human:
                _human.send_card_click('PASS')
            elif _gm and _gm.mouse_over and _human:
                _human.send_card_click(_gm.mouse_over.cards[-1])
        elif event.type == KEYDOWN:
            _gm.keypress(event.key)


def draw():
    global _card_sprites

    card_sprites, other_sprites = _gm.play()
    _card_sprites = card_sprites

    if _pass_btn:
        _pass_btn.highlight(_pass_btn.rect.collidepoint(pygame.mouse.get_pos()))

    under_mouse = [s for s in _card_sprites
                   if s.rect.collidepoint(pygame.mouse.get_pos())]
    _gm.notify_mouseover(under_mouse)

    for position, player in enumerate(_gm.positions):
        for tag in _ui_sprites.sprites():
            if hasattr(tag, 'text') and tag.text[-len(player.name):] == player.name:
                tag.set_text(f'{AbstractPlayer.ranking_names[position]} {player.name}')
                break

    screen.fill(pygame.color.THECOLORS['white'])
    _card_sprites.draw(screen)
    other_sprites.draw(screen)
    _ui_sprites.draw(screen)

    mouse_pos = pygame.mouse.get_pos()
    debug_lines = [
        f'Mouse: {mouse_pos}',
        f'Hit: {[(s.card, s.rect) for s in under_mouse]}' if under_mouse else 'Hit: none',
        f'Highlighted: {_gm.mouse_over}' if _gm.mouse_over else 'Highlighted: none',
    ]
    for i, line in enumerate(debug_lines):
        screen.blit(debug_font.render(line, True, pygame.Color('red')),
                    (5, height - 90 + i * 20))
