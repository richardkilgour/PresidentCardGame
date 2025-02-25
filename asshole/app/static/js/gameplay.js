var socket = io.connect('http://' + document.domain + ':' + location.port);

console.log("GAME JS Found");

// Default values for three opponents: initially zero cards each.
var opponent_cards = [0, 0, 0];

socket.on('connect', function() {
    console.log('WebSocket connection established');
});

document.addEventListener("DOMContentLoaded", function () {
    start_game_button.addEventListener("click", function () {
        socket.emit("start_game");
    })
    start_game_button.disabled = true;
};

function log_out() {
    console.log("Requesting logout");
    socket.emit('logout');
    // Also make HTTP request to ensure session is cleared
    fetch('/logout', {method: 'POST'})
        .then(() => window.location.href = '/');
}


socket.on('notify_current_hand', function(data) {
    console.log(arguments.callee.name);
    playerCards = data.playerCards;
    var handDiv = document.getElementById('player-hand');
    var html = "";
    for (let i = 0; i < playerCards.length; i++) {
        var card = playerCards[i];
        // card[0]: value, card[1]: suit, card[2]: extra info
        html += renderCard(card[0], card[1], i, card[2]);
    }
    handDiv.innerHTML = html;

})


socket.on('notify_opponent_hands', function(data) {
    console.log(arguments.callee.name);
    opponent_cards = data.opponent_cards;

    const opponentContainers = [
        document.getElementById("opponent-1-hand"),
        document.getElementById("opponent-2-hand"),
        document.getElementById("opponent-3-hand")
    ];

    // Clear existing cards before updating
    opponentContainers.forEach(container => {
        if (container) container.innerHTML = "";
    });

    opponent_cards.forEach((cardCount, index) => {
        if (index < opponentContainers.length && opponentContainers[index]) {
            for (let i = 0; i < cardCount; i++) {
                // TODO: Replace 1, "&clubs;" with back of cards
                opponentContainers[index].appendChild(createCard(1, "&clubs;", i, true));
            }
        }
    });

})

socket.on('notify_player_joined', function(data) {
    alert(data.new_player + " joined the game.");
});

socket.on('notify_hand_start', function(data) {
    if (data.starter == player_id) {
        alert("It's your turn to start!");
        // Enable move input and submission here
    }
});

socket.on('notify_hand_won', function(data) {
    alert(data.winner + " won the hand.");
});

socket.on('notify_played_out', function(data) {
    alert(data.opponent + " played out at position " + data.pos);
});

socket.on('notify_play', function(data) {
    alert(data.player + " played " + data.meld);
});

function play_cards(cards) {
    socket.emit('play_cards', {'cards': cards});
}


function leave_game() {
    socket.emit('leave_game');
}

function readyToStart() {
    console.log(arguments.callee.name);
    socket.emit('ready_to_start');
}

function createCard(value, suit, index, playable) {
    const suits = {
        "&clubs;": "clubs",
        "&spades;": "spades",
        "&diams;": "diamonds",
        "&hearts;": "hearts"
    };

    const suitIndex = Object.keys(suits).indexOf(suit);
    const cardId = `${value}_${suitIndex}`;

    const cardHitArea = document.createElement("div");
    cardHitArea.className = "card_hit_area";
    cardHitArea.style.left = `${index}em`;
    cardHitArea.style.top = `${((index - 6) / 4) ** 2}em`;
    cardHitArea.style.transform = `rotate(${7 * (index - 6)}deg)`;

    if (playable) {
        cardHitArea.onmouseover = () => moveCardUp(cardId);
        cardHitArea.onmouseout = () => moveCardDown(cardId);
    }

    const card = document.createElement("div");
    card.className = playable ? "card" : "card_small";
    card.id = cardId;

    const front = document.createElement("div");
    front.className = suit === "&clubs;" || suit === "&spades;" ? "front" : "front red";

    const indexTop = document.createElement("div");
    indexTop.className = "index";
    const indexBottom = document.createElement("div");
    indexBottom.className = "index_bottom";

    if (value === 11) {
        indexTop.innerHTML = `A<br>${suit}`;
        indexBottom.innerHTML = `A<br>${suit}`;
        front.innerHTML += `<div class="ace">${suit}</div>`;
    } else if (value < 8) {
        indexTop.innerHTML = `${value + 3}<br>${suit}`;
        indexBottom.innerHTML = `${value + 3}<br>${suit}`;
    } else if (value === 12) {
        indexTop.innerHTML = `2<br>${suit}`;
        indexBottom.innerHTML = `2<br>${suit}`;
        front.innerHTML += `<div class="spotB1">${suit}</div><div class="spotB5">${suit}</div>`;
    }

    front.appendChild(indexTop);
    front.appendChild(indexBottom);

    // Face cards & Jokers
    if (value >= 8 && value <= 13) {
        const img = document.createElement("img");
        img.className = "face";
        img.width = 80;
        img.height = 130;

        if (value === 8) img.src = `static/img/jack_${suits[suit]}.jpg`;
        else if (value === 9) img.src = `static/img/queen_${suits[suit]}.jpg`;
        else if (value === 10) img.src = `static/img/king_${suits[suit]}.jpg`;
        else if (value === 13) img.src = `static/img/${suit === "&spades;" ? "black_joker" : "red_joker"}.jpg`;

        front.appendChild(img);
    }

    card.appendChild(front);
    cardHitArea.appendChild(card);

    return cardHitArea;
}
