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
        handContainer.appendChild(renderCard(card[0], card[1], index, true, card[2])); // Adjust card data format if needed
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
                opponentContainers[index].appendChild(renderCard(-1, "", i, false, false));
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
    socket.emit("request_game_state");
});

socket.on('notify_hand_won', function(data) {
    alert(data.winner + " won the hand.");
    socket.emit("request_game_state");
});

socket.on('notify_played_out', function(data) {
    alert(data.player + " played out at position " + data.pos);
    socket.emit("request_game_state");
});

socket.on('hand_won', function(data) {
    alert(data.winner + " won the round");
    socket.emit("request_game_state");
});

socket.on('notify_player_turn', function(data) {
    console.log('notify_player_turn: ' + data.player);

    // Remove existing borders
    document.getElementById('playfield_left').style.border = 'none';
    document.getElementById('playfield_center').style.border = 'none';
    document.getElementById('playfield_right').style.border = 'none';
    document.getElementById('playfield_bottom').style.border = 'none';

    // Get the names from the HTML elements
    const opponent1Name = document.getElementById('opponent-1-name').textContent.trim();
    const opponent2Name = document.getElementById('opponent-2-name').textContent.trim();
    const opponent3Name = document.getElementById('opponent-3-name').textContent.trim();
    const playerName = document.getElementById('player_id').textContent.trim();

    // Determine which player's turn it is and highlight the corresponding div
    let playfieldDivId;
    if (data.player === opponent1Name) {
        playfieldDivId = 'playfield_left';
    } else if (data.player === opponent2Name) {
        playfieldDivId = 'playfield_center';
    } else if (data.player === opponent3Name) {
        playfieldDivId = 'playfield_right';
    } else if (data.player === playerName) {
        playfieldDivId = 'playfield_bottom';
    }

    if (playfieldDivId) {
        const playfieldDiv = document.getElementById(playfieldDivId);
        if (playfieldDiv) playfieldDiv.style.border = '5px solid red'; // Big, fat border
    }

    socket.emit("request_game_state");
});

socket.on('card_played', function(data) {
    // Get the names from the HTML elements
    const opponent1Name = document.getElementById('opponent-1-name').textContent.trim();
    const opponent2Name = document.getElementById('opponent-2-name').textContent.trim();
    const opponent3Name = document.getElementById('opponent-3-name').textContent.trim();
    const playerName = document.getElementById('player_id').textContent.trim();

    // Determine the corresponding arena div based on the player name
    let arenaDivId;
    if (data.player_id === opponent1Name) {
        arenaDivId = 'arena_left';
    } else if (data.player_id === opponent2Name) {
        arenaDivId = 'arena_center';
    } else if (data.player_id === opponent3Name) {
        arenaDivId = 'arena_right';
    } else if (data.player_id === playerName) {
        arenaDivId = 'arena_bottom';
    }

    console.log("Card(s) " + data.card_id + " played by " + data.player_id)

    // Get the arena div
    const arenaDiv = document.getElementById(arenaDivId);
    // Clear any existing content in the arena div
    arenaDiv.innerHTML = '';

    // Check if the card_id array is empty
    if (data.card_id.length === 0) {
        // Display "PASSED" message
        const passedElement = document.createElement('h1');
        passedElement.textContent = 'PASSED';
        arenaDiv.appendChild(passedElement);
    } else {
        // Render the card(s)
        data.card_id.forEach((card, index) => {
            const cardElement = renderCard(card[0], card[1], index, false, false);
            arenaDiv.appendChild(cardElement);
        });
    }

    // Emit request for game state
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
    console.log(arguments.callee.name + ": " + cards);
    socket.emit('play_cards', {'cards': cards});
}


function moveCardUp(...cardIds) {
    for (const cardId of cardIds) {
        const card = document.getElementById(cardId);
        if (card) {
            card.style.top = "0%";
            card.style.transform = "translateY(0%)";
            card.onclick = function() {
                play_cards(cardIds);
            }
        }
    }
}

function moveCardDown(...cardIds) {
    for (const cardId of cardIds) {
        const card = document.getElementById(cardId);
        if (card) {
            card.style.top = "100%";
            card.style.transform = "translateY(-100%)";
            card.onClick = undefined;
        }
    }
}

function renderCard(value, suit, index, player_card, playable) {
    const suits = {
        "♠": "spades",
        "♣": "clubs",
        "♦": "diamonds",
        "♥": "hearts"
    };

    const suitIndex = Object.keys(suits).indexOf(suit);

    let cardId;
    if (player_card) {
        cardId = `${value}_${suitIndex}`;
    }
    else {
        // Non-player cards can't be accessed by mouse events, so call the something else
        cardId = `${value}__${suitIndex}`;
    }

    const cardHitArea = document.createElement("div");
    cardHitArea.className = "card_hit_area";
    cardHitArea.style.left = `${index}em`;
    cardHitArea.style.top = `${((index - 6) / 4) ** 2}em`;
    cardHitArea.style.transform = `rotate(${7 * (index - 6)}deg)`;

    if (playable) {
        const playableCards = [cardId];
        for (let i = 0; i <= suitIndex; i++) {
            const similarCardId = `${value}_${i}`;
            const similarCard = document.getElementById(similarCardId);
            if (similarCard) {
                playableCards.push(similarCardId);
            }
        }
        console.log(`Setting mouse events for ${playableCards}:`);
        cardHitArea.onmouseover = () => moveCardUp(...playableCards);
        cardHitArea.onmouseout = () => moveCardDown(...playableCards);
    }

    const card = document.createElement("div");
    card.className = player_card ? "card" : "card_small";
    card.id = cardId;


    const front = document.createElement("div");
    front.className = suit === "♣" || suit === "♠" ? "front" : "front red";

    const indexTop = document.createElement("div");
    indexTop.className = "index";
    const indexBottom = document.createElement("div");
    indexBottom.className = "index_bottom";

    // Draw the Values to the side of the cards

    // Special case for Ace
    if ((value === 11) && (suit != "♠")) {
        indexTop.innerHTML = `A<br>${suit}`;
        indexBottom.innerHTML = `A<br>${suit}`;
    } else if (value < 8) {
        indexTop.innerHTML = `${value + 3}<br>${suit}`;
        indexBottom.innerHTML = `${value + 3}<br>${suit}`;
    } else if (value == 12) {
        indexTop.innerHTML = `2<br>${suit}`;
        indexBottom.innerHTML = `2<br>${suit}`;
    }

    // Draw the pips
    if ((value === 11) && (suit != "♠")) {
        front.innerHTML += `<div class="ace">${suit}</div>`;
    } else if (value == 12) {
        front.innerHTML += `<div class="spotB1">${suit}</div>`;
        front.innerHTML += `<div class="spotB5">${suit}</div>`;
    } else if (value == 0) {
        front.innerHTML += `<div class="spotB1">${suit}</div>`;
        front.innerHTML += `<div class="spotB3">${suit}</div>`;
        front.innerHTML += `<div class="spotB5">${suit}</div>`;
    } else if (value < 8) {
        // Must cards have these pips
        front.innerHTML += `<div class="spotA1">${suit}</div>`;
        front.innerHTML += `<div class="spotA5">${suit}</div>`;
        front.innerHTML += `<div class="spotC1">${suit}</div>`;
        front.innerHTML += `<div class="spotC5">${suit}</div>`;
    }
    // Add any extra pips for cards from 5 to 10 (values 2 to 7)
    if (value == 2) {
        front.innerHTML += `<div class="spotB3">${suit}</div>`;
    } else if (value == 3) {
        front.innerHTML += `<div class="spotA3">${suit}</div>`;
        front.innerHTML += `<div class="spotC3">${suit}</div>`;
    } else if (value == 4) {
        front.innerHTML += `<div class="spotA3">${suit}</div>`;
        front.innerHTML += `<div class="spotC3">${suit}</div>`;
        front.innerHTML += `<div class="spotB2">${suit}</div>`;
    } else if (value == 5) {
        front.innerHTML += `<div class="spotA3">${suit}</div>`;
        front.innerHTML += `<div class="spotC3">${suit}</div>`;
        front.innerHTML += `<div class="spotB2">${suit}</div>`;
        front.innerHTML += `<div class="spotB4">${suit}</div>`;
    } else if (value == 6) {
        front.innerHTML += `<div class="spotA2">${suit}</div>`;
        front.innerHTML += `<div class="spotC2">${suit}</div>`;
        front.innerHTML += `<div class="spotA4">${suit}</div>`;
        front.innerHTML += `<div class="spotC4">${suit}</div>`;
        front.innerHTML += `<div class="spotB3">${suit}</div>`;
    } else if (value == 7) {
        front.innerHTML += `<div class="spotA2">${suit}</div>`;
        front.innerHTML += `<div class="spotC2">${suit}</div>`;
        front.innerHTML += `<div class="spotA4">${suit}</div>`;
        front.innerHTML += `<div class="spotC4">${suit}</div>`;
        front.innerHTML += `<div class="spotB2">${suit}</div>`;
        front.innerHTML += `<div class="spotB4">${suit}</div>`;
    }

    front.appendChild(indexTop);
    front.appendChild(indexBottom);

    // Back-facing cards (value is ignored, but can be black or red)
    if (value < 0) {
        const img = document.createElement("img");
        img.className = "face";
        img.width = 80;
        img.height = 130;
        img.src = `../static/img/${suit === "♠" ? "black" : "red"}_back.jpg`;
        front.appendChild(img);
    }
    // Face cards & Jokers
    if ((value >= 8 && value <= 11) || (value === 13) || ((value === 11) && (suit == "♠"))) {
        const img = document.createElement("img");
        img.className = "face";
        img.width = 80;
        img.height = 130;
        const suitName = suits[suit];
        if (value === 8) img.src = `../static/img/jack_${suits[suit]}.jpg`;
        else if (value === 9) img.src = `../static/img/queen_${suits[suit]}.jpg`;
        else if (value === 10) img.src = `../static/img/king_${suits[suit]}.jpg`;
        else if (value === 13) img.src = `../static/img/${suit === "♠" ? "black_joker" : "red_joker"}.jpg`;
        else img.src = `../static/img/ace_spades.jpg`;
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

    if (playerName) {
        nameElement.textContent = playerName;
        statusElement.textContent = status;
        handElement.innerHTML = ""; // Clear existing cards

        // Add cards if player has joined
        for (let i = 0; i < cardCount; i++) {
            handElement.appendChild(renderCard(-1, "", i, false, false));
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
