from random import shuffle

from asshole.gym_env.Episode import Episode
from asshole.GameMaster import GameMaster
from asshole.player.HTMLPlayer import HTMLPlayer
from asshole.player.PlayerSimple import PlayerSimple
from asshole.player.PlayerHolder import PlayerHolder
from asshole.player.PlayerSplitter import PlayerSplitter
from asshole.player.TensorflowPlayer import TensorflowPlayer
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    # Allow the user to create a new game
    return render_template("index.html")


@app.route('/playfield')
def play_static():
    return render_template("playfield.html")


@app.route('/play', methods = ['POST', 'GET'])
def play():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        gm = GameMaster()
        player_names = ['Sara', 'Richard', 'snAkbar']
        for i in range(0,3):
            select = request.form.get('P'+str(i+1))
            if select == 'easy':
                gm.make_player(PlayerSimple, player_names[i])
            elif select == 'easy':
                gm.make_player(PlayerHolder, player_names[i])
            elif select == 'easy':
                gm.make_player(PlayerSplitter, player_names[i])
            else:
                gm.make_player(TensorflowPlayer, player_names[i])
        gm.make_player(HTMLPlayer, "Silvia")
        shuffle(gm.deck)
        this_episode = Episode(gm.players, gm.positions, gm.deck, gm.listener_list)
        this_episode.deal()
        allCards = []
        # Get cards from the player's perspective
        selection = gm.players[3].possible_plays()
        print(selection)
        for player in gm.players:
            cards = []
            for card in player._hand:
                suit_string = '&spades;'
                if card.suit() == 1:
                    suit_string = '&hearts;'
                elif card.suit() == 2:
                    suit_string = '&diams;'
                elif card.suit() == 3:
                    suit_string = '&clubs;'
                # playable if it's the computer player, it's higher than the meld and maybe there's more than one
                playable = 0
                if player == gm.players[3]:
                    for s in selection:
                        print(s.cards)
                        if s.cards and card == s.cards[-1] and card.suit() == s.cards[-1].suit():
                            print(f'setting card {card} to {len(s.cards)}')
                            playable = len(s.cards)

                cards.append([card.value(), suit_string, playable])
            allCards.append(cards)
        # Append played cards
        allCards.append([[3, '&diams;'], [3, '&hearts;']])
        allCards.append([[5, '&spades;']])
        allCards.append([[6, '&clubs;']])
        allCards.append([[0, '&clubs;']])

        return render_template("playfield_dynamic.html", allCards=allCards)


if __name__ == '__main__':
    app.run(debug=True)
