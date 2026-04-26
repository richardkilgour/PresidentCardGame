#!/usr/bin/env python
"""
Online (server-client) mode for the PyGame UI.

Public API used by __main__:
    connect(url, username, password, is_anon) -> action tuple
    try_restore_session(server_addr)          -> action tuple or None
    draw()                                    -> renders current state
    send_play_meld() / send_pass()            -> fire a move
    send_quit()                               -> leave game (replaced by AI)
    check_replace_click(pos) -> str | None    -> username if Replace button hit
    replace_with_ai(username)                 -> swap disconnected player
    get_client() / get_username() / get_state()
    game_started() -> bool                    -> True once after server signals start
"""
from __future__ import annotations

import os
import threading

import pygame

from president.core.PlayingCard import PlayingCard
from president.ui.PyGame.GuiElements import PlayerNameLabel, PassButton, button_label
from president.ui.PyGame.PyGameCard import PyGameCard
from president.ui.PyGame.app import (
    screen, width, height,
    font_small, font_med,
    find_pos, pygame_player_name,
)

_SESSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.session.json')

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_pycards:             dict[int, PyGameCard] = {}
_state:               dict | None           = None
_client                                     = None   # ServerClient
_username:            str | None            = None
_hover_meld_indices:  set[int]              = set()
_game_started_flag:   bool                  = False

# Disconnection / replacement tracking
_disconnected:        set[str]              = set()
_replace_avail:       set[str]              = set()
_replace_btns:        dict[str, button_label] = {}

_pass_btn  = PassButton(width // 2 - 60, height // 2 - 20)
_quit_btn  = button_label('Leave Game', width - 130, 10, 120, 34, font_size=16)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def get_client():
    return _client

def get_username() -> str | None:
    return _username

def get_state() -> dict | None:
    return _state

def game_started() -> bool:
    """Consume and return the one-shot game-started signal."""
    global _game_started_flag
    if _game_started_flag:
        _game_started_flag = False
        return True
    return False


# ---------------------------------------------------------------------------
# Player actions
# ---------------------------------------------------------------------------

def send_play_meld():
    if not _client or not _hover_meld_indices:
        return
    cards = [
        f"{_pycards[i].card.get_value()}_{_pycards[i].card.get_suit()}"
        for i in _hover_meld_indices
    ]
    _client.play_cards(cards)


def send_pass():
    if _client:
        _client.play_cards('PASSED')


def send_quit():
    """Leave the game; the server replaces us with AI and keeps our slot reserved."""
    if _client:
        _client.quit_game(timeout=5.0)
        _client.disconnect()


def replace_with_ai(username: str):
    if _client:
        _client.replace_with_ai(username)


def check_replace_click(pos) -> str | None:
    """Return the username whose Replace button was clicked, or None."""
    for username, btn in _replace_btns.items():
        if btn.rect.collidepoint(pos):
            return username
    return None


def check_quit_click(pos) -> bool:
    return _quit_btn.rect.collidepoint(pos)


# ---------------------------------------------------------------------------
# Internal callbacks
# ---------------------------------------------------------------------------

def _wire_callbacks(client, *, on_rejoin=None):
    def _on_state(s):
        global _state
        _state = s

    def _on_started():
        global _game_started_flag
        _game_started_flag = True

    def _on_disconnected(username):
        _disconnected.add(username)

    def _on_replace_available(username):
        _replace_avail.add(username)

    def _on_player_replaced(username):
        _disconnected.discard(username)
        _replace_avail.discard(username)

    def _on_player_quit(username):
        _disconnected.discard(username)
        _replace_avail.discard(username)

    client.on_game_state          = _on_state
    client.on_game_started        = _on_started
    client.on_rejoin_game         = on_rejoin
    client.on_player_disconnected = _on_disconnected
    client.on_replace_available   = _on_replace_available
    client.on_player_replaced     = _on_player_replaced
    client.on_player_quit         = _on_player_quit


# ---------------------------------------------------------------------------
# Connect / restore
# ---------------------------------------------------------------------------

def connect(url: str, username: str | None, password: str | None,
            is_anon: bool) -> tuple:
    """
    Log in and connect to *url*.
    Returns ('online', None), ('game_list', games), or ('error', msg).
    """
    global _client, _username, _state

    from president.client.server_client import ServerClient
    client = ServerClient(url)
    try:
        if is_anon or not username:
            uname = client.login_anonymous(name=pygame_player_name)
        else:
            if not client.login(username, password or ''):
                return ('error', '! Login failed — check credentials.')
            uname = username

        rejoin_ev = threading.Event()
        _wire_callbacks(client, on_rejoin=rejoin_ev.set)

        client.connect()
        client.save_session(_SESSION_FILE)
        _client   = client
        _username = uname

        rejoin_ev.wait(timeout=2.0)
        if rejoin_ev.is_set():
            client.request_state(timeout=3.0)
            return ('online', None)

        return ('game_list', client.list_games())

    except Exception as e:
        _client = None
        return ('error', f'! {e}')


def try_restore_session(server_addr: str) -> tuple | None:
    """
    Try to reconnect using a saved session token.
    Returns ('online', None), ('game_list', games), or None if no saved session.
    """
    global _client, _username, _state

    from president.client.server_client import ServerClient
    restored = ServerClient.restore_session(_SESSION_FILE, server_addr)
    if not restored:
        return None

    try:
        rejoin_ev = threading.Event()
        _wire_callbacks(restored, on_rejoin=rejoin_ev.set)
        restored.connect()
        restored.save_session(_SESSION_FILE)
        _client   = restored
        _username = restored.username

        rejoin_ev.wait(timeout=2.0)
        if rejoin_ev.is_set():
            restored.request_state(timeout=3.0)
            return ('online', None)

        return ('game_list', restored.list_games())

    except Exception:
        _client = None
        return None


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _init_pycards():
    global _pycards
    if not _pycards:
        _pycards = {i: PyGameCard(i) for i in range(54)}


def _card_idx(value: int, suit_str: str) -> int:
    return value * 4 + PlayingCard.suit_list.index(suit_str)


def _rebuild_replace_buttons():
    """Rebuild the Replace-with-AI button set from current _replace_avail."""
    global _replace_btns
    _replace_btns = {}
    for i, username in enumerate(sorted(_replace_avail)):
        y = height // 2 + 10 + i * 50
        _replace_btns[username] = button_label(
            f'Replace {username}', width - 210, y, 200, 40, font_size=16)


def _render_state(state: dict):
    """Build sprite groups mirroring PyGameMaster.play() layout."""
    _init_pycards()
    for pc in _pycards.values():
        pc.set_card_params(faceup=False, highlighted=False, newAngle=0)

    visible_cards  = pygame.sprite.Group()
    visible_others = pygame.sprite.Group()

    player_names  = state['player_names']
    player_status = state['player_status']
    player_hand   = state['player_hand']
    opp_counts    = state['opponent_cards']

    used_indices: set[int] = set()
    for value, suit_str, _ in player_hand:
        used_indices.add(_card_idx(value, suit_str))
    for status in player_status:
        if isinstance(status, list):
            for v, s in status:
                used_indices.add(_card_idx(v, s))

    facedown_pool = [i for i in range(54) if i not in used_indices]

    # --- Local player hand (seat 0, bottom) ---
    CARD_RAISE = 20
    mid = len(player_hand) / 2.0
    for j, (value, suit_str, _playable) in enumerate(player_hand):
        card_idx = _card_idx(value, suit_str)
        pc = _pycards[card_idx]
        card_angle = (j - mid) * 6
        pos = find_pos((width // 2, height), 1.5 * pc.height, card_angle)
        highlighted = card_idx in _hover_meld_indices
        pc.set_card_params(faceup=True, highlighted=highlighted, newAngle=-card_angle)
        pc.rect.x = int(pos[0])
        pc.rect.y = int(pos[1]) - (CARD_RAISE if highlighted else 0)
        visible_cards.add(pc)

    # --- Opponent hands face-down (seats 1, 2, 3) ---
    pool_offset = 0
    for opp_i, count in enumerate(opp_counts):
        i = opp_i + 1
        mid_hand = count / 2.0
        for j in range(count):
            if pool_offset >= len(facedown_pool):
                break
            card_idx = facedown_pool[pool_offset]
            pool_offset += 1
            pc = _pycards[card_idx]
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

    # --- Table melds + status labels ---
    meld_xy = [
        (width // 2,          2 * height // 3 - 120),
        (width // 3 - 40,     height // 3),
        (width // 2,          120),
        (2 * width // 3 + 40, height // 3),
    ]
    for i, (status, (mx, my)) in enumerate(zip(player_status, meld_xy)):
        if isinstance(status, list) and status:
            for j, (v, s) in enumerate(status):
                card_idx = _card_idx(v, s)
                pc = _pycards.get(card_idx)
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

    # --- Player name edge labels (mark disconnected players) ---
    name_xy = [
        (width // 2, height - 60),
        (0,          height // 3),
        (width // 2, 0),
        (width - 60, height // 3),
    ]
    for i, (nx, ny) in enumerate(name_xy):
        name = player_names[i] if i < len(player_names) else ''
        label = f'{name} [DC]' if name in _disconnected else name
        visible_others.add(PlayerNameLabel(label, nx, ny))

    return visible_cards, visible_others


# ---------------------------------------------------------------------------
# Main draw
# ---------------------------------------------------------------------------

def draw():
    screen.fill(pygame.Color('white'))
    mouse_pos = pygame.mouse.get_pos()

    is_turn = bool(_state and _state.get('is_my_turn'))

    _hover_meld_indices.clear()
    if is_turn and _pycards and _state:
        hovered_idx = None
        for value, suit_str, _ in reversed(_state.get('player_hand', [])):
            card_idx = _card_idx(value, suit_str)
            pc = _pycards.get(card_idx)
            if pc and pc.rect.collidepoint(mouse_pos):
                hovered_idx = card_idx
                break
        if hovered_idx is not None:
            best: set[int] | None = None
            for meld_cards in _state.get('possible_melds', []):
                indices = {_card_idx(v, s) for v, s in meld_cards}
                if hovered_idx in indices and (best is None or len(indices) > len(best)):
                    best = indices
            if best:
                _hover_meld_indices.update(best)

    if _state:
        card_sprites, other_sprites = _render_state(_state)
        card_sprites.draw(screen)
        other_sprites.draw(screen)
    else:
        waiting = font_med.render('Waiting for game to start…', True, pygame.Color('gray'))
        screen.blit(waiting, (width // 2 - waiting.get_width() // 2, height // 2))

    can_pass = bool(_state and _state.get('can_pass', False))
    if is_turn and can_pass:
        _pass_btn.highlight(_pass_btn.rect.collidepoint(mouse_pos))
        screen.blit(_pass_btn.image, _pass_btn.rect)

    if is_turn:
        ind = font_small.render('▶ Your turn!', True, pygame.Color('darkgreen'))
        screen.blit(ind, (width // 2 - ind.get_width() // 2, height // 2 - 65))

    # Replace-with-AI buttons (visible to all players when a slot is available)
    if _replace_avail:
        _rebuild_replace_buttons()
        for btn in _replace_btns.values():
            btn.render()
            screen.blit(btn.image, btn.rect)

    # Leave Game button (top-right corner)
    _quit_btn.highlight(_quit_btn.rect.collidepoint(mouse_pos))
    _quit_btn.render()
    screen.blit(_quit_btn.image, _quit_btn.rect)
