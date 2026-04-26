#!/usr/bin/env python
"""
Run a PresidentCardGame game with PyGame.

Supports offline (local engine) and online (server client) modes.

Usage:
    python -m president.ui.PyGame.main                         # startup menu
    python -m president.ui.PyGame.main --server HOST:PORT      # connect (anonymous)
    python -m president.ui.PyGame.main --server HOST:PORT \\
                                       --username alice \\
                                       --password secret       # connect as user
"""
from __future__ import annotations

import argparse
import math
import os
import threading

import pygame
import yaml
from pygame.locals import KEYDOWN

from president.core.AbstractPlayer import AbstractPlayer
from president.core.PlayingCard import PlayingCard
from president.ui.PyGame.GuiElements import PlayerNameLabel, PassButton, StatBox, button_label
from president.ui.PyGame.PyGameCard import PyGameCard
from president.ui.PyGame.PyGameMaster import PyGameMaster
from president.players.PyGamePlayer import PyGamePlayer
from president.players.PlayerNaive import PlayerNaive
from president.players.PlayerSimple import PlayerSimple
from president.players.PlayerHolder import PlayerHolder
from president.players.PlayerSplitter import PlayerSplitter
from president.players.ConsolePlayer import ConsolePlayer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = yaml.safe_load(open(config_path))

_pygame_player_name = next(
    (config[k]['name'] for k in ['player1', 'player2', 'player3', 'player4']
     if config[k].get('type') == 'PyGamePlayer'),
    None
)

size = width, height = config['screen']['width'], config['screen']['height']

PLAYER_TYPES = {
    'PyGamePlayer':  PyGamePlayer,
    'PlayerNaive':   PlayerNaive,
    'PlayerSimple':  PlayerSimple,
    'PlayerHolder':  PlayerHolder,
    'PlayerSplitter': PlayerSplitter,
    'ConsolePlayer': ConsolePlayer,
}

# ---------------------------------------------------------------------------
# Game modes
# ---------------------------------------------------------------------------

STARTUP = 'startup'
SERVER_INPUT = 'server_input'
GAME_LIST = 'game_list'
LOBBY = 'lobby'
OFFLINE = 'offline'
ONLINE = 'online'

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
# Geometry helpers (shared with PyGameMaster)
# ---------------------------------------------------------------------------

def find_pos(pos, dist, angle):
    return (pos[0] + math.sin(math.radians(angle)) * dist,
            pos[1] - math.cos(math.radians(angle)) * dist)


# ---------------------------------------------------------------------------
# Simple text-input widget
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


# ---------------------------------------------------------------------------
# Mode: STARTUP
# ---------------------------------------------------------------------------

_btn_offline = button_label('Play Offline',       width // 2 - 120, height // 2 - 60, 240, 50)
_btn_online  = button_label('Connect to Server',  width // 2 - 120, height // 2 + 20, 240, 50)


def draw_startup():
    screen.fill(pygame.Color('white'))
    title = font_large.render('President Card Game', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, height // 3))
    _btn_offline.render()
    screen.blit(_btn_offline.image, _btn_offline.rect)
    _btn_online.render()
    screen.blit(_btn_online.image, _btn_online.rect)


def handle_startup_click(pos):
    if _btn_offline.rect.collidepoint(pos):
        return OFFLINE
    if _btn_online.rect.collidepoint(pos):
        return SERVER_INPUT
    return STARTUP


# ---------------------------------------------------------------------------
# Mode: SERVER_INPUT
# ---------------------------------------------------------------------------

_sv_server   = TextInput(width // 2 - 180, height // 2 - 90, 360, 36, placeholder='localhost:5000')
_sv_username = TextInput(width // 2 - 180, height // 2 - 30, 360, 36, placeholder='username (blank = anonymous)')
_sv_password = TextInput(width // 2 - 180, height // 2 + 30, 360, 36, placeholder='password', password=True)
_sv_inputs   = [_sv_server, _sv_username, _sv_password]

_btn_connect = button_label('Connect',   width // 2 - 130, height // 2 + 90, 120, 40)
_btn_anon    = button_label('Anonymous', width // 2 +  10, height // 2 + 90, 120, 40)
_btn_back_sv = button_label('Back',      20, 20, 80, 34, font_size=18)

_sv_status = ''  # feedback message


def draw_server_input():
    screen.fill(pygame.Color('white'))
    title = font_med.render('Connect to Server', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, height // 2 - 150))

    for label_text, widget in [
        ('Server:', _sv_server),
        ('Username:', _sv_username),
        ('Password:', _sv_password),
    ]:
        lbl = font_small.render(label_text, True, pygame.Color('black'))
        screen.blit(lbl, (width // 2 - 240, widget.rect.y + 8))
        widget._render()
        screen.blit(widget.image, widget.rect)

    for btn in (_btn_connect, _btn_anon, _btn_back_sv):
        btn.render()
        screen.blit(btn.image, btn.rect)

    if _sv_status:
        col = pygame.Color('red') if _sv_status.startswith('!') else pygame.Color('darkgreen')
        msg = font_small.render(_sv_status.lstrip('!'), True, col)
        screen.blit(msg, (width // 2 - msg.get_width() // 2, height // 2 + 145))


# ---------------------------------------------------------------------------
# Mode: GAME_LIST
# ---------------------------------------------------------------------------

_gl_game_btns: list[tuple[dict, button_label]] = []
_btn_create   = button_label('Create New Game', width // 2 - 120, 0, 240, 40)  # y set dynamically
_btn_back_gl  = button_label('Back', 20, 20, 80, 34, font_size=18)
_gl_status    = ''


def _rebuild_game_list_buttons(games: list):
    global _gl_game_btns
    _gl_game_btns = []
    y = 120
    for g in games:
        players_str = ', '.join(g['players'])
        label = f'{g["id"][:8]}…  ({players_str})'
        btn = button_label(label, width // 2 - 220, y, 440, 36, font_size=18)
        _gl_game_btns.append((g, btn))
        y += 50
    _btn_create.y = y + 10
    _btn_create.render()


def draw_game_list():
    screen.fill(pygame.Color('white'))
    title = font_med.render('Ongoing Games', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, 70))

    for _, btn in _gl_game_btns:
        btn.render()
        screen.blit(btn.image, btn.rect)

    if not _gl_game_btns:
        msg = font_small.render('No games found.', True, pygame.Color('gray'))
        screen.blit(msg, (width // 2 - msg.get_width() // 2, 140))

    _btn_create.render()
    screen.blit(_btn_create.image, (_btn_create.x, _btn_create.y))
    _btn_back_gl.render()
    screen.blit(_btn_back_gl.image, _btn_back_gl.rect)

    if _gl_status:
        col = pygame.Color('red') if _gl_status.startswith('!') else pygame.Color('darkgreen')
        msg = font_small.render(_gl_status.lstrip('!'), True, col)
        screen.blit(msg, (width // 2 - msg.get_width() // 2, height - 40))


# ---------------------------------------------------------------------------
# Mode: LOBBY
# ---------------------------------------------------------------------------

_lobby_players: list[str] = []    # player names currently in game
_lobby_is_owner: bool = False     # True when we created this game
_lobby_ai_added: list[bool] = [False, False, False]  # seats 1/2/3 filled
_AI_DIFFICULTIES = ['Easy', 'Medium', 'Hard']
_lobby_ai_diff   = ['Medium', 'Medium', 'Medium']    # chosen difficulty per seat

_btn_start_game  = button_label('Start Game', width // 2 - 100, height - 100, 200, 44)
_btn_back_lobby  = button_label('Back',       20, 20, 80, 34, font_size=18)
_lobby_add_btns: list[button_label] = []  # one per seat 1-3
_lobby_diff_btns: list[button_label] = []
_lobby_status = ''


def _rebuild_lobby_buttons():
    global _lobby_add_btns, _lobby_diff_btns
    _lobby_add_btns  = []
    _lobby_diff_btns = []
    for i in range(3):
        y = 160 + i * 70
        added = _lobby_ai_added[i]
        label = f'Seat {i + 1}: {_lobby_players[i + 1] if len(_lobby_players) > i + 1 else "Empty"}'
        if not added:
            add_lbl = f'+ Add AI ({_lobby_ai_diff[i]})'
        else:
            add_lbl = f'✓ {_lobby_players[i + 1] if len(_lobby_players) > i + 1 else "AI"}'
        _lobby_add_btns.append(button_label(add_lbl, width // 2 + 20, y, 200, 36, font_size=18))
        diff_lbl = _lobby_ai_diff[i]
        _lobby_diff_btns.append(button_label(diff_lbl, width // 2 + 230, y, 80, 36, font_size=16))


def draw_lobby():
    screen.fill(pygame.Color('white'))
    title = font_med.render('Game Lobby', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, 30))

    you_lbl = font_small.render(f'You: {_online_username or "?"}', True, pygame.Color('black'))
    screen.blit(you_lbl, (width // 2 - you_lbl.get_width() // 2, 75))

    _rebuild_lobby_buttons()
    for i, (add_btn, diff_btn) in enumerate(zip(_lobby_add_btns, _lobby_diff_btns)):
        y = 160 + i * 70
        seat_lbl = font_small.render(f'Seat {i + 1}:', True, pygame.Color('black'))
        screen.blit(seat_lbl, (width // 2 - 200, y + 8))
        if _lobby_is_owner:
            add_btn.render()
            screen.blit(add_btn.image, add_btn.rect)
            if not _lobby_ai_added[i]:
                diff_btn.render()
                screen.blit(diff_btn.image, diff_btn.rect)

    _btn_start_game.render()
    screen.blit(_btn_start_game.image, _btn_start_game.rect)
    _btn_back_lobby.render()
    screen.blit(_btn_back_lobby.image, _btn_back_lobby.rect)

    if _lobby_status:
        col = pygame.Color('red') if _lobby_status.startswith('!') else pygame.Color('darkgreen')
        msg = font_small.render(_lobby_status.lstrip('!'), True, col)
        screen.blit(msg, (width // 2 - msg.get_width() // 2, height - 50))


# ---------------------------------------------------------------------------
# Offline mode  (unchanged logic, re-packaged into functions)
# ---------------------------------------------------------------------------

_offline_gm: PyGameMaster | None = None
_offline_card_sprites  = pygame.sprite.Group()
_offline_ui_sprites    = pygame.sprite.Group()
_offline_human: PyGamePlayer | None = None
_offline_pass_btn: PassButton | None = None
_offline_stat_box: StatBox | None   = None


def _init_offline():
    global _offline_gm, _offline_card_sprites, _offline_ui_sprites
    global _offline_human, _offline_pass_btn, _offline_stat_box

    _offline_gm = PyGameMaster(width, height)
    _offline_card_sprites  = pygame.sprite.Group()
    _offline_ui_sprites    = pygame.sprite.Group()

    players = [config['player1'], config['player2'], config['player3'], config['player4']]
    # Rotate so the PyGamePlayer is always index 0 (bottom of screen)
    for _ in range(len(players)):
        if players[0]['type'] == 'PyGamePlayer':
            break
        players.append(players.pop(0))

    for i, p in enumerate(players):
        player_class = PLAYER_TYPES[p['type']]
        _offline_gm.make_player(player_class, p['name'])
        if i == 0:
            _offline_ui_sprites.add(PlayerNameLabel(p['name'], width // 2, height - 60))
        elif i == 1:
            _offline_ui_sprites.add(PlayerNameLabel(p['name'], 0, height // 3))
        elif i == 2:
            _offline_ui_sprites.add(PlayerNameLabel(p['name'], width // 2, 0))
        elif i == 3:
            _offline_ui_sprites.add(PlayerNameLabel(p['name'], width - 60, height // 3))

    _offline_human = _offline_gm.get_human_player()
    _offline_pass_btn = PassButton(width // 2, height // 2)
    _offline_ui_sprites.add(_offline_pass_btn)
    _offline_stat_box = StatBox(0, height - 150)
    _offline_ui_sprites.add(_offline_stat_box)


def update_offline(events, mouse_pos):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if _offline_pass_btn and _offline_pass_btn.rect.collidepoint(event.pos) and _offline_human:
                _offline_human.send_card_click('PASS')
            elif _offline_gm and _offline_gm.mouse_over and _offline_human:
                _offline_human.send_card_click(_offline_gm.mouse_over.cards[-1])
        elif event.type == KEYDOWN:
            _offline_gm.keypress(event.key)


def draw_offline():
    global _offline_card_sprites

    card_sprites, other_sprites = _offline_gm.play()
    _offline_card_sprites = card_sprites

    if _offline_pass_btn:
        _offline_pass_btn.highlight(_offline_pass_btn.rect.collidepoint(pygame.mouse.get_pos()))

    under_mouse = [s for s in _offline_card_sprites
                   if s.rect.collidepoint(pygame.mouse.get_pos())]
    _offline_gm.notify_mouseover(under_mouse)

    for position, player in enumerate(_offline_gm.positions):
        for tag in _offline_ui_sprites.sprites():
            if hasattr(tag, 'text') and tag.text[-len(player.name):] == player.name:
                tag.set_text(f'{AbstractPlayer.ranking_names[position]} {player.name}')
                break

    screen.fill(pygame.color.THECOLORS['white'])
    _offline_card_sprites.draw(screen)
    other_sprites.draw(screen)
    _offline_ui_sprites.draw(screen)

    mouse_pos = pygame.mouse.get_pos()
    debug_lines = [
        f'Mouse: {mouse_pos}',
        f'Hit: {[(s.card, s.rect) for s in under_mouse]}' if under_mouse else 'Hit: none',
        f'Highlighted: {_offline_gm.mouse_over}' if _offline_gm.mouse_over else 'Highlighted: none',
    ]
    for i, line in enumerate(debug_lines):
        screen.blit(debug_font.render(line, True, pygame.Color('red')),
                    (5, height - 90 + i * 20))


_SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.session.json')


def _wire_online_callbacks(client, on_rejoin=None):
    def _on_state(s):
        global _online_state
        _online_state = s
    def _on_joined(name):
        global _lobby_players
        _lobby_players = list(set(_lobby_players) | {name})
    def _on_started():
        global mode
        mode = ONLINE
    client.on_game_state    = _on_state
    client.on_player_joined = _on_joined
    client.on_game_started  = _on_started
    client.on_rejoin_game   = on_rejoin


# ---------------------------------------------------------------------------
# Online mode
# ---------------------------------------------------------------------------

# Pool of 54 PyGameCard sprites for online rendering (lazy-initialised)
_online_pycards_by_index: dict[int, PyGameCard] = {}
_online_state: dict | None = None
_online_client = None                 # ServerClient instance
_online_username: str | None = None

_online_pass_btn         = PassButton(width // 2 - 60, height // 2 - 20)
_online_hover_meld_indices: set[int] = set()  # indices of cards in the currently hovered meld


def _init_online_pycards():
    global _online_pycards_by_index
    if not _online_pycards_by_index:
        _online_pycards_by_index = {i: PyGameCard(i) for i in range(54)}


def _card_index_from_server(value: int, suit_str: str) -> int:
    return value * 4 + PlayingCard.suit_list.index(suit_str)


def _render_online_state(state: dict):
    """Build sprite groups mirroring PyGameMaster.play() layout exactly."""
    _init_online_pycards()

    for pc in _online_pycards_by_index.values():
        pc.set_card_params(faceup=False, highlighted=False, newAngle=0)

    visible_cards  = pygame.sprite.Group()
    visible_others = pygame.sprite.Group()

    player_names  = state['player_names']
    player_status = state['player_status']
    player_hand   = state['player_hand']    # [[value, suit_str, playable], ...]
    opp_counts    = state['opponent_cards'] # [n1, n2, n3]

    # Collect indices of cards already placed face-up (hand + table melds)
    used_indices: set[int] = set()
    for value, suit_str, _ in player_hand:
        used_indices.add(_card_index_from_server(value, suit_str))
    for status in player_status:
        if isinstance(status, list):
            for v, s in status:
                used_indices.add(_card_index_from_server(v, s))

    # Remaining cards are distributed face-down to opponents
    facedown_pool = [i for i in range(54) if i not in used_indices]

    # --- Local player hand (position 0, bottom of screen) ---
    CARD_RAISE = 20
    mid = len(player_hand) / 2.0
    for j, (value, suit_str, _playable) in enumerate(player_hand):
        card_idx = _card_index_from_server(value, suit_str)
        pc = _online_pycards_by_index[card_idx]
        card_angle = (j - mid) * 6
        pos = find_pos((width // 2, height), 1.5 * pc.height, card_angle)
        highlighted = card_idx in _online_hover_meld_indices
        pc.set_card_params(faceup=True, highlighted=highlighted, newAngle=-card_angle)
        pc.rect.x = int(pos[0])
        pc.rect.y = int(pos[1]) - (CARD_RAISE if highlighted else 0)
        visible_cards.add(pc)

    # --- Opponent hands face-down (positions 1, 2, 3) ---
    pool_offset = 0
    for opp_i, count in enumerate(opp_counts):
        i = opp_i + 1
        mid_hand = count / 2.0
        for j in range(count):
            if pool_offset >= len(facedown_pool):
                break
            card_idx = facedown_pool[pool_offset]
            pool_offset += 1
            pc = _online_pycards_by_index[card_idx]
            card_angle = i * 90 + (j - mid_hand) * 6
            if i == 1:
                base_pos = (-50, height // 3)
            elif i == 2:
                base_pos = (width // 2, -pc.height - 20)
            else:
                base_pos = (width, height // 3)
            pos = find_pos(base_pos, 1.5 * pc.height, card_angle)
            pc.set_card_params(faceup=False, newAngle=-card_angle)
            pc.rect.x = int(pos[0])
            pc.rect.y = int(pos[1])
            visible_cards.add(pc)

    # --- Table melds + status labels (positions match PyGameMaster.set_label_pos) ---
    meld_xy = [
        (width // 2,          2 * height // 3 - 120),
        (width // 3 - 40,     height // 3),
        (width // 2,          120),
        (2 * width // 3 + 40, height // 3),
    ]
    for i, (status, (mx, my)) in enumerate(zip(player_status, meld_xy)):
        if isinstance(status, list) and status:
            for j, (v, s) in enumerate(status):
                card_idx = _card_index_from_server(v, s)
                pc = _online_pycards_by_index.get(card_idx)
                if pc is None:
                    continue
                pc.set_card_params(faceup=True, newAngle=i * 90)
                if i in (0, 2):
                    pc.rect.x = mx + 40 * j
                    pc.rect.y = my
                else:
                    pc.rect.x = mx
                    pc.rect.y = my + j * height // 12
                visible_cards.add(pc)
        else:
            name = player_names[i] if i < len(player_names) else ''
            if status == 'Passed':
                lbl_text = f'{name} PASSED'
            elif status in (None, 'Waiting', '␆'):
                lbl_text = f'{name} WAITING'
            elif isinstance(status, str):
                lbl_text = f'{name}: {status}'
            else:
                lbl_text = name
            visible_others.add(PlayerNameLabel(lbl_text, mx, my))

    # --- Player name labels at screen edges (same as _init_offline) ---
    name_xy = [
        (width // 2, height - 60),
        (0,          height // 3),
        (width // 2, 0),
        (width - 60, height // 3),
    ]
    for i, (nx, ny) in enumerate(name_xy):
        name = player_names[i] if i < len(player_names) else ''
        visible_others.add(PlayerNameLabel(name, nx, ny))

    return visible_cards, visible_others


def send_online_play_meld():
    """Play the currently highlighted meld immediately."""
    if not _online_client or not _online_hover_meld_indices:
        return
    cards = [
        f"{_online_pycards_by_index[i].card.get_value()}_{_online_pycards_by_index[i].card.get_suit()}"
        for i in _online_hover_meld_indices
    ]
    _online_client.play_cards(cards)


def send_online_pass():
    if not _online_client:
        return
    _online_client.play_cards('PASSED')


def draw_online():
    screen.fill(pygame.Color('white'))
    mouse_pos = pygame.mouse.get_pos()

    is_turn = bool(_online_state and _online_state.get('is_my_turn'))

    # Recompute hover meld each frame (uses rects from previous frame — 1-frame lag is fine)
    _online_hover_meld_indices.clear()
    if is_turn and _online_pycards_by_index and _online_state:
        hovered_idx = None
        for value, suit_str, _ in reversed(_online_state.get('player_hand', [])):
            card_idx = _card_index_from_server(value, suit_str)
            pc = _online_pycards_by_index.get(card_idx)
            if pc and pc.rect.collidepoint(mouse_pos):
                hovered_idx = card_idx
                break
        if hovered_idx is not None:
            # Pick the largest valid meld that contains this card
            best: set[int] | None = None
            for meld_cards in _online_state.get('possible_melds', []):
                indices = {_card_index_from_server(v, s) for v, s in meld_cards}
                if hovered_idx in indices and (best is None or len(indices) > len(best)):
                    best = indices
            if best:
                _online_hover_meld_indices.update(best)

    if _online_state:
        card_sprites, other_sprites = _render_online_state(_online_state)
        card_sprites.draw(screen)
        other_sprites.draw(screen)
    else:
        waiting = font_med.render('Waiting for game to start…', True, pygame.Color('gray'))
        screen.blit(waiting, (width // 2 - waiting.get_width() // 2, height // 2))

    can_pass = bool(_online_state and _online_state.get('can_pass', False))
    if is_turn and can_pass:
        _online_pass_btn.highlight(_online_pass_btn.rect.collidepoint(mouse_pos))
        screen.blit(_online_pass_btn.image, _online_pass_btn.rect)

    if is_turn:
        ind = font_small.render('▶ Your turn!', True, pygame.Color('darkgreen'))
        screen.blit(ind, (width // 2 - ind.get_width() // 2, height // 2 - 65))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global mode
    global _sv_status, _gl_status, _lobby_status
    global _online_client, _online_username, _online_state
    global _lobby_is_owner

    # ---- Parse CLI args ----
    parser = argparse.ArgumentParser(description='President Card Game (PyGame)')
    parser.add_argument('--server',   type=str, metavar='HOST[:PORT]',
                        help='Skip startup menu and connect to this server.')
    parser.add_argument('--username', type=str, help='Username for online play.')
    parser.add_argument('--password', type=str, help='Password for online play.')
    args = parser.parse_args()

    # If --server given on CLI, pre-fill server_input and jump straight there
    if args.server:
        _sv_server.text = args.server
        if args.username:
            _sv_username.text = args.username
        if args.password:
            _sv_password.text = args.password
        mode = SERVER_INPUT

        # Try to restore a saved session and reconnect automatically
        from president.client.server_client import ServerClient
        restored = ServerClient.restore_session(_SESSION_FILE, args.server)
        if restored:
            rejoin_ev = threading.Event()
            _wire_online_callbacks(restored, on_rejoin=rejoin_ev.set)
            try:
                restored.connect()
                restored.save_session(_SESSION_FILE)
                _online_client   = restored
                _online_username = restored.username
                rejoin_ev.wait(timeout=2.0)
                if rejoin_ev.is_set():
                    restored.request_state(timeout=3.0)
                    mode = ONLINE
                else:
                    games = restored.list_games()
                    _rebuild_game_list_buttons(games)
                    mode = GAME_LIST
            except Exception:
                _online_client = None
    else:
        mode = STARTUP

    clock = pygame.time.Clock()
    running = True

    while running:
        events = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()

        # ---- Global quit ----
        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # ==================================================================
        # STARTUP
        # ==================================================================
        if mode == STARTUP:
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mode = handle_startup_click(event.pos)
                    if mode == OFFLINE:
                        _init_offline()
            draw_startup()

        # ==================================================================
        # SERVER INPUT
        # ==================================================================
        elif mode == SERVER_INPUT:
            for event in events:
                for inp in _sv_inputs:
                    inp.handle_event(event)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if _btn_back_sv.rect.collidepoint(pos):
                        mode = STARTUP
                        _sv_status = ''
                    elif _btn_connect.rect.collidepoint(pos) or _btn_anon.rect.collidepoint(pos):
                        anon = _btn_anon.rect.collidepoint(pos)
                        _sv_status = 'Connecting…'
                        draw_server_input()
                        pygame.display.flip()

                        from president.client.server_client import ServerClient
                        url = _sv_server.text.strip() or 'localhost:5000'
                        if '://' not in url:
                            url = f'http://{url}'

                        client = ServerClient(url)
                        try:
                            if anon or not _sv_username.text.strip():
                                uname = client.login_anonymous(name=_pygame_player_name)
                            else:
                                ok = client.login(_sv_username.text.strip(),
                                                  _sv_password.text.strip())
                                if not ok:
                                    _sv_status = '! Login failed — check credentials.'
                                    client = None
                                    continue
                                uname = _sv_username.text.strip()

                            rejoin_ev = threading.Event()
                            _wire_online_callbacks(client, on_rejoin=rejoin_ev.set)

                            client.connect()
                            client.save_session(_SESSION_FILE)
                            _online_client   = client
                            _online_username = uname

                            rejoin_ev.wait(timeout=2.0)
                            if rejoin_ev.is_set():
                                client.request_state(timeout=3.0)
                                mode = ONLINE
                                _sv_status = ''
                            else:
                                games = client.list_games()
                                _rebuild_game_list_buttons(games)
                                mode = GAME_LIST
                                _sv_status = ''

                        except Exception as e:
                            _sv_status = f'! {e}'
                            _online_client = None

            draw_server_input()

        # ==================================================================
        # GAME LIST
        # ==================================================================
        elif mode == GAME_LIST:
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if _btn_back_gl.rect.collidepoint(pos):
                        mode = SERVER_INPUT
                        _gl_status = ''
                    elif _btn_create.rect.collidepoint(pos):
                        game_id = _online_client.create_game(timeout=5.0)
                        if game_id:
                            _lobby_players = [_online_username or '']
                            _lobby_ai_added[:] = [False, False, False]
                            _lobby_is_owner = True
                            mode = LOBBY
                            _gl_status = ''
                        else:
                            _gl_status = '! Failed to create game.'
                    else:
                        for g, btn in _gl_game_btns:
                            if btn.rect.collidepoint(pos):
                                _online_client.join_game(g['id'])
                                _lobby_players = list(g['players'])
                                _lobby_ai_added[:] = [True, True, True]
                                _lobby_is_owner = False
                                mode = LOBBY
                                _gl_status = ''
                                break
            draw_game_list()

        # ==================================================================
        # LOBBY
        # ==================================================================
        elif mode == LOBBY:
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if _btn_back_lobby.rect.collidepoint(pos):
                        mode = GAME_LIST
                        games = _online_client.list_games()
                        _rebuild_game_list_buttons(games)
                        _lobby_status = ''
                    elif _btn_start_game.rect.collidepoint(pos):
                        _lobby_status = 'Starting…'
                        draw_lobby()
                        pygame.display.flip()
                        ok = _online_client.start_game(timeout=10.0)
                        if ok:
                            mode = ONLINE
                            _lobby_status = ''
                        else:
                            _lobby_status = '! Start timed out (game may start anyway).'
                    else:
                        # Add AI / cycle difficulty buttons
                        for i, (add_btn, diff_btn) in enumerate(
                                zip(_lobby_add_btns, _lobby_diff_btns)):
                            if add_btn.rect.collidepoint(pos) and not _lobby_ai_added[i]:
                                seat = i + 1
                                name = f'CPU-{seat}'
                                diff = _lobby_ai_diff[i]
                                _online_client.add_ai_player(seat, name, diff)
                                _lobby_ai_added[i] = True
                                if len(_lobby_players) <= seat:
                                    _lobby_players.append(name)
                                else:
                                    _lobby_players[seat] = name
                            elif diff_btn.rect.collidepoint(pos) and not _lobby_ai_added[i]:
                                # Cycle difficulty
                                cur = _AI_DIFFICULTIES.index(_lobby_ai_diff[i])
                                _lobby_ai_diff[i] = _AI_DIFFICULTIES[(cur + 1) % 3]
            draw_lobby()

        # ==================================================================
        # OFFLINE
        # ==================================================================
        elif mode == OFFLINE:
            if _offline_gm is None:
                _init_offline()
            update_offline(events, mouse_pos)
            draw_offline()

        # ==================================================================
        # ONLINE
        # ==================================================================
        elif mode == ONLINE:
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    is_my_turn = bool(_online_state and _online_state.get('is_my_turn'))
                    if (_online_pass_btn.rect.collidepoint(pos)
                            and _online_state
                            and _online_state.get('can_pass')
                            and is_my_turn):
                        send_online_pass()
                    elif is_my_turn and _online_hover_meld_indices:
                        send_online_play_meld()

            draw_online()

        pygame.display.flip()
        clock.tick(60)

    if _online_client:
        _online_client.disconnect()
    pygame.quit()


if __name__ == '__main__':
    main()
