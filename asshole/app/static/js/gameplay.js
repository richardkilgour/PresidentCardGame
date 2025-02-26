var socket = io.connect('http://' + document.domain + ':' + location.port);

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
    socket.emit("request_game_state");
});

function log_out() {
    console.log("Requesting logout");
    socket.emit('logout');
    // Also make HTTP request to ensure session is cleared
    fetch('/logout', {method: 'POST'})
        .then(() => window.location.href = '/');
}

function update_state(blah) {
    document.getElementById('debug_area').innerHTML = blah
}

socket.on("current_game_state", function (data) {
    console.log("Received full game state:", data);

    // Create an array of all existing players (opponents + self)
    let existingPlayers = data.opponent_details
        .map(opponent => opponent.name)
        .filter(name => name !== null); // Remove empty slots

    console.log("Updated existingPlayers list:", existingPlayers);

    // Count how many opponents have joined (ignoring null slots)
    let opponentCount = data.opponent_details.filter(opponent => opponent.name !== null).length;

    // Enable "Start Game" button if 3 opponents have joined
    let startGameButton = document.getElementById("start_game_button");
    startGameButton.disabled = opponentCount < 3;  // Disable unless 3+ players

    // Update opponent slots dynamically
    data.opponent_details.forEach((opponent, index) => {
        updateOpponentSlot(index + 1, opponent.name, opponent.card_count, opponent.status, data.is_owner, existingPlayers);
    });

    // Update player's hand
    updatePlayerHand(data.player_hand);
});

// Function to update player's hand UI
function updatePlayerHand(cards) {
    const handContainer = document.getElementById("player-hand");
    handContainer.innerHTML = ""; // Clear previous cards

    cards.forEach((card, index) => {
        handContainer.appendChild(renderCard(card[0], card[1], index, true)); // Adjust card data format if needed
    });
}


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
                opponentContainers[index].appendChild(renderCard(-1, "&clubs;", i, false));
            }
        }
    });

})

socket.on('notify_player_joined', function(data) {
    socket.emit("request_game_state");
});

socket.on('notify_game_started', function(data) {
    alert("game_started");
    socket.emit("request_game_state");
});

socket.on('notify_hand_start', function(data) {
    alert("It's turn for " + data.starter + " to start!");
    socket.emit("request_game_state");
});

socket.on('notify_hand_won', function(data) {
    alert(data.winner + " won the hand.");
    socket.emit("request_game_state");
});

socket.on('notify_played_out', function(data) {
    alert(data.opponent + " played out at position " + data.pos);
    socket.emit("request_game_state");
});

socket.on('card_played', function(data) {
    // TODO: Display the card
    alert(data.player_id + " played " + data.card_id);
    socket.emit("request_game_state");
});

function leave_game() {
    socket.emit('leave_game');
}

function readyToStart() {
    console.log(arguments.callee.name);
    socket.emit('start_game');
}

function play_cards(cards) {
    console.log(arguments.callee.name);
    socket.emit('play_cards', {'cards': cards});
}


function moveCardUp() {
  for (var i = 0; i < arguments.length; i++) {
      x = arguments[i];
      card = document.getElementById(x);
      card.style.top = "0%";
      card.style.transform = "translateY(0%)";
      card.onclick = function() {
        play_cards(x);
      }
  }
}

function moveCardDown() {
  for (var i = 0; i < arguments.length; i++) {
      x = arguments[i];
      card = document.getElementById(x);
      card.style.top = "100%";
      card.style.transform = "translateY(-100%)";
      card.onClick = undefined;
  }
}

function renderCard(value, suit, index, playable) {
    const suits = {
        "♣": "clubs",
        "♠": "spades",
        "♦": "diamonds",
        "♥": "hearts"
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
    front.className = suit === "♣" || suit === "♠" ? "front" : "front red";

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
        const suitName = suits[suit];
        if (value === 8) img.src = `../static/img/jack_${suits[suit]}.jpg`;
        else if (value === 9) img.src = `../static/img/queen_${suits[suit]}.jpg`;
        else if (value === 10) img.src = `../static/img/king_${suits[suit]}.jpg`;
        else if (value === 13) img.src = `../static/img/${suit === "♠" ? "black_joker" : "red_joker"}.jpg`;
        front.appendChild(img);
    }
    // Back-facing cards
    if (value < 0) {
        const img = document.createElement("img");
        img.className = "face";
        img.width = 80;
        img.height = 130;
        img.src = `../static/img/red_back.jpg`;
        front.appendChild(img);
    }

    card.appendChild(front);
    cardHitArea.appendChild(card);

    return cardHitArea;
}

const AI_OPPONENTS = [
    "Alexi (Easy)", "Jordan (Easy)", "Kai (Easy)",
    "Samara (Medium)", "Quincy (Medium)", "Tian (Medium)",
    "Eshaan (Hard)", "Amari (Hard)", "Riven (Hard)"
];

function populateAIDropdown(slotIndex, existingPlayers) {
    const aiDropdown = document.getElementById(`ai-opponent-${slotIndex}`);

    // Clear existing options
    aiDropdown.innerHTML = "";

    // Debugging: Log existingPlayers to see if it's defined
    console.log(`Populating AI Dropdown for slot ${slotIndex}:`, existingPlayers);

    // Ensure existingPlayers is an array
    if (!Array.isArray(existingPlayers)) {
        console.warn(`existingPlayers is not an array! Received:`, existingPlayers);
        existingPlayers = [];  // Default to an empty array
    }

    // Filter out AI names that are already taken
    const availableAIs = AI_OPPONENTS.filter(ai => !existingPlayers.includes(ai.split(" ")[0]));

    // Populate dropdown
    availableAIs.forEach(ai => {
        let option = document.createElement("option");
        option.value = ai.split(" ")[0];  // Extract the first name
        option.textContent = ai;
        aiDropdown.appendChild(option);
    });
}


// Function to update opponent slot dynamically
function updateOpponentSlot(index, playerName, cardCount, status, isOwner, existingPlayers) {
    const nameElement = document.getElementById(`opponent-${index}-name`);
    const statusElement = document.getElementById(`opponent-${index}-status`);
    const handElement = document.getElementById(`opponent-${index}-hand`);
    const aiSelectElement = document.getElementById(`opponent-${index}-ai-select`);

    console.log("updateOpponentSlot " + index);


    if (playerName) {
        nameElement.textContent = playerName;
        statusElement.textContent = status;
        handElement.innerHTML = ""; // Clear existing cards

        // Add cards if player has joined
        for (let i = 0; i < cardCount; i++) {
            handElement.appendChild(renderCard(-1, "&clubs;", i, false));
        }

        // Hide AI selection (owner cannot add AI to an occupied slot)
        if (aiSelectElement) {
            aiSelectElement.style.display = "none";
        }
    } else {
        // Player has not joined
        nameElement.textContent = "Waiting...";
        statusElement.textContent = "Waiting for Player to join";
        handElement.innerHTML = ""; // No cards shown
        console.log("isOwner = " + isOwner);
        console.log("aiSelectElement = " + aiSelectElement);
        // Show AI selection if the game owner is viewing
        if (isOwner && aiSelectElement) {
            aiSelectElement.style.display = "block";
            populateAIDropdown(index, existingPlayers);
        }
    }
}

// Called when the owner selects an AI player
// Called when the owner selects an AI player
function addAIPlayer(opponentIndex) {
    const aiDropdown = document.getElementById(`ai-opponent-${opponentIndex}`);
    const aiName = aiDropdown.value;

    // Find the full entry in AI_OPPONENTS array
    const aiEntry = AI_OPPONENTS.find(entry => entry.startsWith(aiName));

    // Extract difficulty from the parentheses (e.g., "Easy", "Medium", "Hard")
    const aiDifficulty = aiEntry.match(/\((.*?)\)/)[1];

    console.log(`Adding AI Player: ${aiName}, Difficulty: ${aiDifficulty}`);

    socket.emit("add_ai_player", { opponentIndex, aiName, aiDifficulty });
}
