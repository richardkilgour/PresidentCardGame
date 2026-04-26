#!/usr/bin/env python
"""
PyGame UI entry point.

Usage:
    python -m president.ui.PyGame                                      # startup menu
    python -m president.ui.PyGame --server HOST:PORT                   # connect (anon)
    python -m president.ui.PyGame --server HOST:PORT \\
                                  --username alice --password s3cr3t
"""
from __future__ import annotations

import argparse

import pygame

from president.ui.PyGame import menus, offline, online
from president.ui.PyGame.app import (
    screen,
    STARTUP, SERVER_INPUT, GAME_LIST, LOBBY, OFFLINE, ONLINE,
)


def main():
    parser = argparse.ArgumentParser(description='President Card Game (PyGame)')
    parser.add_argument('--server',   type=str, metavar='HOST[:PORT]')
    parser.add_argument('--username', type=str)
    parser.add_argument('--password', type=str)
    args = parser.parse_args()

    if args.server:
        menus.prefill_server(args.server, args.username or '', args.password or '')
        result = online.try_restore_session(args.server)
        mode = _apply_connect_result(*result) if result else SERVER_INPUT
    else:
        mode = STARTUP

    clock   = pygame.time.Clock()
    running = True

    while running:
        events    = pygame.event.get()
        mouse_pos = pygame.mouse.get_pos()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        # ------------------------------------------------------------------
        if mode == STARTUP:
            action = menus.handle_startup(events)
            if action:
                if action[0] == 'offline':
                    offline.init()
                    mode = OFFLINE
                elif action[0] == 'server_input':
                    mode = SERVER_INPUT
            menus.draw_startup()

        # ------------------------------------------------------------------
        elif mode == SERVER_INPUT:
            action = menus.handle_server_input(events)
            if action:
                if action[0] == 'back':
                    menus.set_sv_status('')
                    mode = STARTUP
                elif action[0] == 'connect':
                    _, url, username, password, is_anon = action
                    menus.set_sv_status('Connecting…')
                    menus.draw_server_input()
                    pygame.display.flip()
                    result = online.connect(url, username, password, is_anon)
                    mode = _apply_connect_result(*result)
                    if mode == SERVER_INPUT:
                        menus.set_sv_status(result[1])  # error message
            menus.draw_server_input()

        # ------------------------------------------------------------------
        elif mode == GAME_LIST:
            action = menus.handle_game_list(events)
            if action:
                if action[0] == 'back':
                    menus.set_gl_status('')
                    mode = SERVER_INPUT
                elif action[0] == 'create':
                    game_id = online.get_client().create_game(timeout=5.0)
                    if game_id:
                        menus.enter_lobby([online.get_username() or ''], is_owner=True)
                        menus.set_gl_status('')
                        mode = LOBBY
                    else:
                        menus.set_gl_status('! Failed to create game.')
                elif action[0] == 'join':
                    g = action[1]
                    online.get_client().join_game(g['id'])
                    menus.enter_lobby(list(g['players']), is_owner=False)
                    menus.set_gl_status('')
                    mode = LOBBY
            menus.draw_game_list()

        # ------------------------------------------------------------------
        elif mode == LOBBY:
            # Server may start the game independently (e.g. another player is owner)
            if online.game_started():
                menus.set_lobby_status('')
                mode = ONLINE
            else:
                action = menus.handle_lobby(events, online.get_client())
                if action:
                    if action[0] == 'back_to_list':
                        games = online.get_client().list_games()
                        menus.rebuild_game_list(games)
                        menus.set_lobby_status('')
                        mode = GAME_LIST
                    elif action[0] == 'start':
                        menus.set_lobby_status('Starting…')
                        menus.draw_lobby(online.get_username())
                        pygame.display.flip()
                        ok = online.get_client().start_game(timeout=10.0)
                        if ok:
                            menus.set_lobby_status('')
                            mode = ONLINE
                        else:
                            menus.set_lobby_status('! Start timed out (game may start anyway).')
            menus.draw_lobby(online.get_username())

        # ------------------------------------------------------------------
        elif mode == OFFLINE:
            offline.handle(events, mouse_pos)
            offline.draw()

        # ------------------------------------------------------------------
        elif mode == ONLINE:
            state = online.get_state()
            for event in events:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    online.send_quit()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    is_my_turn = bool(state and state.get('is_my_turn'))
                    if online.check_quit_click(event.pos):
                        online.send_quit()
                        running = False
                    elif (online._pass_btn.rect.collidepoint(event.pos)
                            and state and state.get('can_pass')
                            and is_my_turn):
                        online.send_pass()
                    elif is_my_turn and online._hover_meld_indices:
                        online.send_play_meld()
                    else:
                        target = online.check_replace_click(event.pos)
                        if target:
                            online.replace_with_ai(target)
            online.draw()

        pygame.display.flip()
        clock.tick(60)

    if online.get_client():
        online.get_client().disconnect()
    pygame.quit()


def _apply_connect_result(action: str, payload) -> str:
    """Translate a connect/restore result into a mode constant."""
    if action == 'online':
        return ONLINE
    if action == 'game_list':
        menus.rebuild_game_list(payload)
        menus.set_gl_status('')
        return GAME_LIST
    # 'error' — stay on SERVER_INPUT; caller sets the status message
    return SERVER_INPUT


if __name__ == '__main__':
    main()
