import os
import sched
import threading
import time

from president.app.game_keeper import GamesKeeper

_INTERVAL = float(os.environ.get('PRESIDENT_SCHEDULER_INTERVAL', '1'))


def step_games(sc):
    from president.app.extensions import socketio
    from president.app.session_manager import user_socket_map

    now = time.time()
    finished_games = []

    for game_id, game_wrapper in list(GamesKeeper().get_games().items()):
        # Advance game logic, respecting per-game speed setting
        if game_wrapper.episode and (now - game_wrapper.last_step_at) >= game_wrapper.step_interval:
            game_wrapper.last_step_at = now
            done = game_wrapper.step()
            if done:
                finished_games.append(game_id)

        # Fire disconnect timeouts
        for username, info in list(game_wrapper.disconnect_info.items()):
            if not info['notified'] and (now - info['time']) >= info['timeout']:
                info['notified'] = True
                if username.endswith(' (unregistered)'):
                    # Anonymous player — auto-replace so the game continues,
                    # but keep slot reserved so they can rejoin
                    game_wrapper.replace_human_with_ai(username, reserved=True)
                    game_wrapper.clear_disconnect(username)
                    socketio.emit('player_replaced', {'username': username}, room=game_id)
                else:
                    socketio.emit(
                        'replace_available',
                        {'username': username, 'game_id': game_id},
                        room=game_id,
                    )

    # Clean up games that have run their full course (no humans asked to end them)
    for game_id in finished_games:
        from president.app.game_persistence import delete_game
        delete_game(game_id)
        GamesKeeper().remove_game(game_id)

    sc.enter(_INTERVAL, 1, step_games, (sc,))


def start_scheduler():
    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(_INTERVAL, 1, step_games, (scheduler,))
    thread = threading.Thread(target=scheduler.run, daemon=True)
    thread.start()
    return thread
