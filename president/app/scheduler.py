import sched
import threading
import time

from president.app.game_keeper import GamesKeeper


def step_games(sc):
    games = GamesKeeper().get_games()
    for game_id, game_wrapper in games.items():
        if game_wrapper.episode:
            game_wrapper.step()
    sc.enter(1, 1, step_games, (sc,))


def start_scheduler():
    scheduler = sched.scheduler(time.time, time.sleep)
    scheduler.enter(1, 1, step_games, (scheduler,))
    thread = threading.Thread(target=scheduler.run, daemon=True)
    thread.start()
    return thread
