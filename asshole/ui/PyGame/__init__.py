#!/usr/bin/env python
"""
Run an Asshole game with PyGame

This module has the main game loop, but he game logic is delegated to the PyGameMaster
Config is read from config.yaml
THIS guy should hold all the game objects
"""
import sys
import yaml
import pygame
from asshole.player.AbstractPlayer import AbstractPlayer
from pygame.locals import *
# TODO: load dynamically using importlib
from AssholeUI.GuiElements import PlayerNameLabel, PassButton, StatBox
from AssholeUI.PyGameCard import PyGameCard
from AssholeUI.PyGameMaster import PyGameMaster
from AssholeUI.PyGamePlayer import PyGamePlayer

from asshole.player.PlayerSimple import PlayerSimple
from asshole.player.PlayerHolder import PlayerHolder
from asshole.player.PlayerSplitter import PlayerSplitter

# TODO: read a configuration file
config = yaml.safe_load(open("./config.yaml"))

size = width, height = config['screen']['width'], config['screen']['height']

pygame.init()
screen = pygame.display.set_mode(size)
# Read config file
# Make some actors
actors = []

turn_count = 0
segment_count = 0

card_sprites_list = pygame.sprite.Group()
ui_sprites_list = pygame.sprite.Group()

TEST_CARD_LAYOUT = False
if TEST_CARD_LAYOUT:
    for i in range(0, 54):
        # pos = (width * random(), height * random())
        row_length = 12
        col_height = 5
        pos = ((width / row_length) * (i % row_length), (height / col_height) * (i // row_length))
        new_card = PyGameCard(i % 4, i // 4, pos)
        new_card.rect.x = pos[0]
        new_card.rect.y = pos[1]
        card_sprites_list.add(new_card)

# PyGame will ceed control to the GM
gm = PyGameMaster(width, height)

players = [config['player1'], config['player2'], config['player3'], config['player4'], ]
# From now on, the human player will be in position 0. 1 is to the left, 2 is opposite and 3 is to the right
p0 = players[-1]
while players[0] != p0:
    if players[0]['type'] == 'PyGamePlayer':
        break
    players.append(players.pop(0))

print(f'{players}')

for i, p in enumerate(players):
    player_class = getattr(sys.modules[__name__], p['type'])
    gm.make_player(player_class, p['name'])
    if p['type'] == 'PyGamePlayer':
        human_player = gm.players[-1]
        assert i == 0
    if i == 0:
        # Yuck - explicitly place them
        ui_sprites_list.add(PlayerNameLabel(p['name'],   width // 2, height - 60))
    elif i == 1:
        ui_sprites_list.add(PlayerNameLabel(p['name'],  0,  height // 3))
    elif i == 2:
        ui_sprites_list.add(PlayerNameLabel(p['name'], width // 2,  0))
    elif i == 3:
        ui_sprites_list.add(PlayerNameLabel(p['name'],  width - 60, height // 3))

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

        # we press a button
        if event.type == KEYDOWN:
            gm.keypress(event.key)
        # where is the mouse
        for s in ui_sprites_list:
            if s == pass_button:
                if s.rect.collidepoint(pygame.mouse.get_pos()):
                    if event.type == pygame.MOUSEBUTTONDOWN:
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

    # increment the current game state, and render sprites
    card_sprites_list, other_sprite_list = gm.play()

    # Add pass and quit buttons
    for position, player in enumerate(gm.positions):
        # Find the nametag
        for tag in ui_sprites_list.sprites():
            if tag.text[-len(player.name):] == player.name:
                break
        # Update the name tags
        tag.set_text(f'{AbstractPlayer.ranking_names[position]} {player.name}')

    # Update the display
    screen.fill(pygame.color.THECOLORS['white'])
    card_sprites_list.draw(screen)
    other_sprite_list.draw(screen)
    ui_sprites_list.draw(screen)
    pygame.display.flip()
    pygame.time.wait(1)
