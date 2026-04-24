import sched
import threading
import time

from president.app.game_keeper import GamesKeeper


def step_games(sc):
    from president.app.extensions import socketio
    from president.app.session_manager import user_socket_map

    now = time.time()
    finished_games = []

    for game_id, game_wrapper in list(GamesKeeper().get_games().items()):
        # Advance game logic
        if game_wrapper.episode:
            done = game_wrapper.step()
            if done:
                finished_games.append(game_id)

        # Fire disconnect timeouts
        for username, info in list(game_wrapper.disconnect_info.items()):
            if not info['notified'] and (now - info['time']) >= info['timeout']:
                info['notified'] = True
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

    sc.enter(1, 1, step_games, (sc,))


def start_scheduler():
    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(1, 1, step_games, (scheduler,))
    thread = threading.Thread(target=scheduler.run, daemon=True)
    thread.start()
    return thread
