#!/usr/bin/env python
"""
Run a PresidentCardGame game with PyGame.

Main game loop — delegates game logic to PyGameMaster.
Config is read from config.yaml in the same directory.
Run with: python -m president.ui.PyGame.main
"""
import os
import yaml
import pygame
from pygame.locals import KEYDOWN

from president.core.AbstractPlayer import AbstractPlayer
from president.ui.PyGame.GuiElements import PlayerNameLabel, PassButton, StatBox
from president.ui.PyGame.PyGameCard import PyGameCard
from president.ui.PyGame.PyGameMaster import PyGameMaster
from president.players.PyGamePlayer import PyGamePlayer
from president.players.PlayerNaive import PlayerNaive
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSplitter import PlayerSplitter
from president.players.ConsolePlayer import ConsolePlayer

PLAYER_TYPES = {
    'PyGamePlayer':  PyGamePlayer,
    'PlayerNaive':   PlayerNaive,
    'PlayerSimple':  PlayerSimple,
    'PlayerHolder':  PlayerHolder,
    'PlayerSplitter': PlayerSplitter,
    'ConsolePlayer': ConsolePlayer,
}

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = yaml.safe_load(open(config_path))

size = width, height = config['screen']['width'], config['screen']['height']

pygame.init()
screen = pygame.display.set_mode(size)

card_sprites_list = pygame.sprite.Group()
ui_sprites_list = pygame.sprite.Group()

gm = PyGameMaster(width, height)

players = [config['player1'], config['player2'], config['player3'], config['player4']]
# Rotate so the PyGamePlayer is always at index 0 (bottom of screen)
p0 = players[-1]
while players[0] != p0:
    if players[0]['type'] == 'PyGamePlayer':
        break
    players.append(players.pop(0))

human_player = None
for i, p in enumerate(players):
    player_class = PLAYER_TYPES[p['type']]
    gm.make_player(player_class, p['name'])
    if p['type'] == 'PyGamePlayer':
        human_player = gm.player_manager.players[-1]
        assert i == 0
    if i == 0:
        ui_sprites_list.add(PlayerNameLabel(p['name'], width // 2, height - 60))
    elif i == 1:
        ui_sprites_list.add(PlayerNameLabel(p['name'], 0, height // 3))
    elif i == 2:
        ui_sprites_list.add(PlayerNameLabel(p['name'], width // 2, 0))
    elif i == 3:
        ui_sprites_list.add(PlayerNameLabel(p['name'], width - 60, height // 3))

clock = pygame.time.Clock()
running = True

pass_button = PassButton(width // 2, height // 2)
ui_sprites_list.add(pass_button)
last_mouse_pos = None

stat_box = StatBox(0, height - 150)
ui_sprites_list.add(stat_box)

while running:

    mouse_is_over = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == KEYDOWN:
            gm.keypress(event.key)

        for s in ui_sprites_list:
            if s == pass_button:
                if s.rect.collidepoint(pygame.mouse.get_pos()):
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if human_player:
                            human_player.send_card_click('PASS')
                    else:
                        s.highlight(True)
                else:
                    s.highlight(False)

        for s in card_sprites_list:
            if s.rect.collidepoint(pygame.mouse.get_pos()):
                last_mouse_pos = pygame.mouse.get_pos()
                gm.notify_mouseover(s)
                mouse_is_over = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    gm.notify_click(s.card)

    if not mouse_is_over and last_mouse_pos != pygame.mouse.get_pos():
        gm.notify_mouseover(None)

    card_sprites_list, other_sprite_list = gm.play()

    for position, player in enumerate(gm.positions):
        for tag in ui_sprites_list.sprites():
            if hasattr(tag, 'text') and tag.text[-len(player.name):] == player.name:
                break
        tag.set_text(f'{AbstractPlayer.ranking_names[position]} {player.name}')

    screen.fill(pygame.color.THECOLORS['white'])
    card_sprites_list.draw(screen)
    other_sprite_list.draw(screen)
    ui_sprites_list.draw(screen)
    pygame.display.flip()
    pygame.time.wait(1)
