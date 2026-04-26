#!/usr/bin/env python
"""
Menu screens for the PyGame UI: STARTUP, SERVER_INPUT, GAME_LIST, LOBBY.

Each handle_*() function processes events and returns an action tuple or None.
Each draw_*() function renders the current screen state.

Action tuples:
    ('offline',      None)
    ('server_input', None)
    ('back',         None)
    ('connect',      url, username_or_none, password_or_none, is_anon)
    ('create',       None)
    ('join',         game_dict)
    ('start',        None)
    ('back_to_list', None)
"""
from __future__ import annotations

import pygame

from president.ui.PyGame.app import (
    screen, width, height,
    font_large, font_med, font_small,
    TextInput,
)
from president.ui.PyGame.GuiElements import button_label

# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------

_btn_offline = button_label('Play Offline',      width // 2 - 120, height // 2 - 60, 240, 50)
_btn_online  = button_label('Connect to Server', width // 2 - 120, height // 2 + 20, 240, 50)


def draw_startup():
    screen.fill(pygame.Color('white'))
    title = font_large.render('President Card Game', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, height // 3))
    _btn_offline.render()
    screen.blit(_btn_offline.image, _btn_offline.rect)
    _btn_online.render()
    screen.blit(_btn_online.image, _btn_online.rect)


def handle_startup(events):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if _btn_offline.rect.collidepoint(event.pos):
                return ('offline', None)
            if _btn_online.rect.collidepoint(event.pos):
                return ('server_input', None)
    return None


# ---------------------------------------------------------------------------
# SERVER_INPUT
# ---------------------------------------------------------------------------

_sv_server   = TextInput(width // 2 - 180, height // 2 - 90, 360, 36,
                         placeholder='localhost:5000')
_sv_username = TextInput(width // 2 - 180, height // 2 - 30, 360, 36,
                         placeholder='username (blank = anonymous)')
_sv_password = TextInput(width // 2 - 180, height // 2 + 30, 360, 36,
                         placeholder='password', password=True)
_sv_inputs   = [_sv_server, _sv_username, _sv_password]

_btn_connect = button_label('Connect',   width // 2 - 130, height // 2 + 90, 120, 40)
_btn_anon    = button_label('Anonymous', width // 2 +  10, height // 2 + 90, 120, 40)
_btn_back_sv = button_label('Back',      20, 20, 80, 34, font_size=18)

_sv_status = ''


def set_sv_status(msg: str):
    global _sv_status
    _sv_status = msg


def prefill_server(server: str = '', username: str = '', password: str = ''):
    """Pre-populate SERVER_INPUT fields (e.g. from CLI args)."""
    if server:   _sv_server.text   = server
    if username: _sv_username.text = username
    if password: _sv_password.text = password


def draw_server_input():
    screen.fill(pygame.Color('white'))
    title = font_med.render('Connect to Server', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, height // 2 - 150))

    for label_text, widget in [
        ('Server:',   _sv_server),
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


def handle_server_input(events):
    for event in events:
        for inp in _sv_inputs:
            inp.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if _btn_back_sv.rect.collidepoint(pos):
                return ('back', None)
            is_anon = _btn_anon.rect.collidepoint(pos)
            if is_anon or _btn_connect.rect.collidepoint(pos):
                url = _sv_server.text.strip() or 'localhost:5000'
                if '://' not in url:
                    url = f'http://{url}'
                return ('connect', url,
                        _sv_username.text.strip() or None,
                        _sv_password.text.strip() or None,
                        is_anon)
    return None


# ---------------------------------------------------------------------------
# GAME_LIST
# ---------------------------------------------------------------------------

_gl_game_btns: list[tuple[dict, button_label]] = []
_btn_create   = button_label('Create New Game', width // 2 - 120, 0, 240, 40)
_btn_back_gl  = button_label('Back', 20, 20, 80, 34, font_size=18)
_gl_status    = ''


def set_gl_status(msg: str):
    global _gl_status
    _gl_status = msg


def rebuild_game_list(games: list):
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


def handle_game_list(events):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if _btn_back_gl.rect.collidepoint(pos):
                return ('back', None)
            if _btn_create.rect.collidepoint(pos):
                return ('create', None)
            for g, btn in _gl_game_btns:
                if btn.rect.collidepoint(pos):
                    return ('join', g)
    return None


# ---------------------------------------------------------------------------
# LOBBY
# ---------------------------------------------------------------------------

_lobby_players:  list[str]  = []
_lobby_is_owner: bool       = False
_lobby_ai_added: list[bool] = [False, False, False]
_AI_DIFFICULTIES             = ['Easy', 'Medium', 'Hard']
_lobby_ai_diff:  list[str]  = ['Medium', 'Medium', 'Medium']

_btn_start_game  = button_label('Start Game', width // 2 - 100, height - 100, 200, 44)
_btn_back_lobby  = button_label('Back',       20, 20, 80, 34, font_size=18)
_lobby_add_btns:  list[button_label] = []
_lobby_diff_btns: list[button_label] = []
_lobby_status = ''


def set_lobby_status(msg: str):
    global _lobby_status
    _lobby_status = msg


def enter_lobby(players: list[str], *, is_owner: bool):
    """Set lobby state when entering from game list (create or join)."""
    global _lobby_players, _lobby_is_owner
    _lobby_players  = list(players)
    _lobby_is_owner = is_owner
    _lobby_ai_added[:] = [False, False, False]
    _lobby_ai_diff[:]  = ['Medium', 'Medium', 'Medium']


def _rebuild_lobby_buttons():
    global _lobby_add_btns, _lobby_diff_btns
    _lobby_add_btns  = []
    _lobby_diff_btns = []
    for i in range(3):
        y = 160 + i * 70
        if _lobby_ai_added[i]:
            add_lbl = f'✓ {_lobby_players[i + 1] if len(_lobby_players) > i + 1 else "AI"}'
        else:
            add_lbl = f'+ Add AI ({_lobby_ai_diff[i]})'
        _lobby_add_btns.append(
            button_label(add_lbl, width // 2 + 20, y, 200, 36, font_size=18))
        _lobby_diff_btns.append(
            button_label(_lobby_ai_diff[i], width // 2 + 230, y, 80, 36, font_size=16))


def draw_lobby(username: str | None = None):
    screen.fill(pygame.Color('white'))
    title = font_med.render('Game Lobby', True, pygame.Color('black'))
    screen.blit(title, (width // 2 - title.get_width() // 2, 30))

    you_lbl = font_small.render(f'You: {username or "?"}', True, pygame.Color('black'))
    screen.blit(you_lbl, (width // 2 - you_lbl.get_width() // 2, 75))

    _rebuild_lobby_buttons()
    for i, (add_btn, diff_btn) in enumerate(zip(_lobby_add_btns, _lobby_diff_btns)):
        seat_lbl = font_small.render(f'Seat {i + 1}:', True, pygame.Color('black'))
        screen.blit(seat_lbl, (width // 2 - 200, 160 + i * 70 + 8))
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


def handle_lobby(events, client):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if _btn_back_lobby.rect.collidepoint(pos):
                return ('back_to_list', None)
            if _btn_start_game.rect.collidepoint(pos):
                return ('start', None)
            for i, (add_btn, diff_btn) in enumerate(zip(_lobby_add_btns, _lobby_diff_btns)):
                if add_btn.rect.collidepoint(pos) and not _lobby_ai_added[i]:
                    seat = i + 1
                    name = f'CPU-{seat}'
                    client.add_ai_player(seat, name, _lobby_ai_diff[i])
                    _lobby_ai_added[i] = True
                    if len(_lobby_players) <= seat:
                        _lobby_players.append(name)
                    else:
                        _lobby_players[seat] = name
                elif diff_btn.rect.collidepoint(pos) and not _lobby_ai_added[i]:
                    cur = _AI_DIFFICULTIES.index(_lobby_ai_diff[i])
                    _lobby_ai_diff[i] = _AI_DIFFICULTIES[(cur + 1) % 3]
    return None
