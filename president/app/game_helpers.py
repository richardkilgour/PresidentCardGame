from flask_socketio import join_room

from president.app.game_event_handler import cards_to_list
from president.app.game_keeper import GamesKeeper
from president.app.session_manager import user_socket_map
from president.players.AsyncPlayer import AsyncPlayer


def _translate_player_status(raw):
    """Translate core game status to a clean app-level value.

    Core types never leak past this point into the API response.
    """
    if isinstance(raw, str):
        return "Waiting" if raw == '␆' else raw
    # Meld object: empty means the player passed, otherwise return card list
    return "Passed" if not raw.cards else cards_to_list(raw.cards)


def add_human_player(user_id, game_id):
    player = AsyncPlayer(user_id)
    for sid in user_socket_map[user_id]:
        join_room(game_id, sid)
    GamesKeeper().add_player(game_id, player)
    print(f"User {user_id} joined game {game_id}")


def find_valid_game(user_id, game_id=None):
    games = GamesKeeper().get_games()

    if not game_id or game_id not in games:
        game_id = next(
            (gid for gid, gm in games.items() if user_id in GamesKeeper().get_player_names(gid)),
            None
        )

    if not game_id or game_id not in games:
        return None

    return game_id


def get_game_state(game_id):
    gm = GamesKeeper().get_game(game_id)
    player_names = GamesKeeper().get_player_names(game_id)
    players = gm.player_manager.players

    if gm.episode:
        player_positions = [player.name if player else -1 for player in gm.positions]
        raw_status = [gm.episode.current_melds[i] if player else "Absent"
                      for i, player in enumerate(players)]
    else:
        player_positions = [-1] * 4
        raw_status = ["Waiting" if player else "Absent" for player in players]

    player_cards = [player._hand if player else [] for player in players]
    player_status = [_translate_player_status(s) for s in raw_status]

    position_icons = ["👑", "🥈", "🥉", "💩"]
    player_stats = []
    for player in players:
        if player:
            current_pos = gm.positions.index(player) if player in gm.positions else None
            player_stats.append({
                "name": player.name,
                "score": player.get_score(),
                "max_consecutive_president": player.max_consecutive_president,
                "current_position": current_pos,
                "current_position_icon": position_icons[current_pos] if current_pos is not None else None,
            })

    stats = {
        "rounds_played": gm.round_number,
        "high_score": gm.high_score,
        "low_score": gm.low_score,
        "players": player_stats,
    }

    return {
        "game_id": game_id,
        "player_names": player_names,
        "player_status": player_status,
        "player_positions": player_positions,
        "player_hands": player_cards,
        "owner": players[0].name,
        "stats": stats,
    }


def get_state_for_user(user_id, game_id=None):
    """Return game state from the perspective of the given user."""
    if not user_id:
        return None

    game_id = find_valid_game(user_id, game_id)
    if not game_id:
        return None

    game_state = get_game_state(game_id)

    if user_id not in game_state["player_names"]:
        raise KeyError

    player_index = game_state["player_names"].index(user_id)

    player_names = game_state["player_names"][player_index:] + game_state["player_names"][:player_index]
    player_status = game_state["player_status"][player_index:] + game_state["player_status"][:player_index]
    player_positions = game_state["player_positions"]

    opponent_cards = []
    for i in range(1, 4):
        opponent_index = (i + player_index) % 4
        opponent_cards.append(len(game_state["player_hands"][opponent_index]))

    game = GamesKeeper().get_game(game_id)
    player = game.player_manager.players[player_index]
    open_card_index = game.open_card_index
    playable_indices = [c.cards[-1].get_index() for c in player.possible_plays(player.target_meld, open_card_index) if c.cards]

    playable_cards = []
    for card in game_state["player_hands"][player_index]:
        playable = card.get_index() in playable_indices
        playable_cards.append([card.get_value(), card.suit_str(), playable])

    is_my_turn = (
        game.episode is not None and
        bool(game.episode.active_players) and
        game.episode.active_players[0].name == user_id
    )

    return {
        "game_id": game_id,
        "player_names": player_names,
        "player_status": player_status,
        "player_positions": player_positions,
        "player_hand": playable_cards,
        "opponent_cards": opponent_cards,
        "is_owner": (game_state["owner"] == user_id),
        "is_my_turn": is_my_turn,
        "stats": game_state["stats"],
    }
