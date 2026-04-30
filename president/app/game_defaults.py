import uuid

_DEFAULT_CONFIGS = [
    # (seed_label, [(ai_name, difficulty_key), ...])  — slots 1-3; slot 0 left open for human
    ("easy",  [("ECHO-9 (AI)", "simple"),   ("NULL-0 (AI)", "simple"),   ("WREN-X (AI)", "simple")]),
    ("mixed", [("WREN-X (AI)", "simple"),   ("ECHO-9 (AI)", "holder"),   ("APEX-1 (AI)", "splitter")]),
    ("hard",  [("VERA-7 (AI)", "splitter"), ("KAEL-3 (AI)", "splitter"), ("APEX-1 (AI)", "splitter")]),
]

_AI_CLASSES = {
    "simple":   lambda: __import__('president.players.PlayerSimple',  fromlist=['PlayerSimple']).PlayerSimple,
    "holder":   lambda: __import__('president.players.PlayerHolder',  fromlist=['PlayerHolder']).PlayerHolder,
    "splitter": lambda: __import__('president.players.PlayerSplitter', fromlist=['PlayerSplitter']).PlayerSplitter,
}


def _make_ai(name: str, difficulty_key: str):
    cls = _AI_CLASSES[difficulty_key]()
    return cls(name)


def seed_default_games() -> None:
    """Ensure one seeded waiting lobby game exists for each difficulty tier."""
    from president.app.extensions import socketio
    from president.app.game_event_handler import GameEventHandler
    from president.app.game_keeper import GamesKeeper
    from president.app.game_wrapper import GameWrapper

    keeper = GamesKeeper()
    existing_labels = {
        game.seed_label
        for game in keeper.get_games().values()
        if game.is_seeded and not game.episode
    }

    created = False
    for label, ai_roster in _DEFAULT_CONFIGS:
        if label in existing_labels:
            continue
        game_id = str(uuid.uuid4())
        game = GameWrapper(game_id, GameEventHandler(socketio, game_id))
        game.is_seeded = True
        game.seed_label = label
        for slot, (ai_name, diff_key) in enumerate(ai_roster, start=1):
            game.add_player(_make_ai(ai_name, diff_key), slot)
        keeper.add_game(game_id, game)
        created = True

    if created:
        socketio.emit('update_game_list', {"games": keeper.game_list()})
