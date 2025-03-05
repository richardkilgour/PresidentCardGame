// Establish namespace to avoid global variables
const CardGame = {
    socket: null,
    opponentCards: [0, 0, 0],
    aiOpponents: [
        "Alexi (Easy)", "Jordan (Easy)", "Kai (Easy)",
        "Samara (Medium)", "Quincy (Medium)", "Tian (Medium)",
        "Eshaan (Hard)", "Amari (Hard)", "Riven (Hard)"
    ],
    suits: {
        "♠": "spades",
        "♣": "clubs",
        "♦": "diamonds",
        "♥": "hearts"
    },

    init: function() {
        this.socket = io.connect('http://' + document.domain + ':' + location.port);
        this.setupSocketEvents();
        const startGameButton = document.getElementById("start_game_button");
        startGameButton.addEventListener("click", () => this.socket.emit("start_game"));
        startGameButton.disabled = true;
        this.socket.emit("request_game_state");
    },

    setupSocketEvents: function() {
        this.socket.on('connect', () => console.log('WebSocket connection established'));
        this.socket.on("current_game_state", (data) => this.handleGameState(data));
        this.socket.on('notify_player_joined', () => this.socket.emit("request_game_state"));
        this.socket.on('notify_game_started', () => {
            alert("game_started");
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_hand_start', () => this.socket.emit("request_game_state"));
        this.socket.on('notify_hand_won', (data) => {
            alert(data.winner + " won the hand.");
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_played_out', (data) => {
            alert(data.player + " played out at position " + data.pos);

            const rankSymbols = ["👑", "🥈", "🥉", "💩"];

            if (data.pos == 0) {
                // Clear existing ranks from all player elements
                document.querySelectorAll("[data-rank]").forEach(el => {
                    el.removeAttribute("data-rank");
                });
            }

            let playerElement
            if (document.getElementById('opponent-1-name').textContent.trim() == data.player) {
                playerElement = document.getElementById('opponent-1-name');
            }
            else if (document.getElementById('opponent-2-name').textContent.trim() == data.player) {
                playerElement = document.getElementById('opponent-2-name');
            }
            else if (document.getElementById('opponent-3-name').textContent.trim() == data.player) {
                playerElement = document.getElementById('opponent-3-name');
            }
            else if (document.getElementById('player_id').textContent.trim() == data.player) {
                playerElement = document.getElementById('player_id');
            }

            if (playerElement) {
                playerElement.setAttribute("data-rank", rankSymbols[data.pos]);
            }
            this.setStatus(data.player, 'Finished');
            this.socket.emit("request_game_state");
        });
        this.socket.on('hand_won', (data) => {
            alert(data.winner + " won the round");
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_player_turn', (data) => this.highlightCurrentPlayerTurn(data));
        this.socket.on('card_played', (data) => {
            this.setPlayedCard(data.player_id, data.card_id);
            this.socket.emit("request_game_state");
        });
    },


    enableStartButton: function() {
        console.log("Enable start button");
        let startGameButton = document.getElementById("start_game_button");
        startGameButton.disabled = false;

        // Add class to trigger animation
        startGameButton.classList.add("fill");

        // Auto-click after 5 seconds
        setTimeout(() => {
            startGameButton.click();
        }, 5000);
    },


    handleGameState: function(data) {
        //console.log("Received full game state:", data);

        // Create an array of all existing players (opponents + self)
        let existingPlayers = data.opponent_details
            .map(opponent => opponent.name)
            .filter(name => name !== null); // Remove empty slots

        // Count how many opponents have joined (ignoring null slots)
        let opponentCount = data.opponent_details.filter(opponent => opponent.name !== null).length;

        // Enable "Start Game" button if 3 opponents have joined
        let startGameButton = document.getElementById("start_game_button");
        if (opponentCount >= 3 && startGameButton.disabled) {
            this.enableStartButton();
        }

        // Update opponent slots dynamically
        data.opponent_details.forEach((opponent, index) => {
            this.updateOpponentSlot(index + 1, opponent.name, opponent.card_count,
                                  opponent.status, data.is_owner, existingPlayers);
        });

        // Update player's hand
        this.updatePlayerHand(data.player_hand);
    },

    updatePlayerHand: function(cards) {
        const handContainer = document.getElementById("player-hand");
        handContainer.innerHTML = ""; // Clear previous cards
        cards.forEach((card, i) => {
            let index = i - Math.floor(cards.length / 2);
            handContainer.appendChild(this.renderCard(card[0], card[1], index, true, card[2]));
        });
    },


    highlightCurrentPlayerTurn: function(data) {
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

        this.socket.emit("request_game_state");
    },

    findArena: function(playerId) {
        // Get the names from the HTML elements
        const opponent1Name = document.getElementById('opponent-1-name').textContent.trim();
        const opponent2Name = document.getElementById('opponent-2-name').textContent.trim();
        const opponent3Name = document.getElementById('opponent-3-name').textContent.trim();
        const playerName = document.getElementById('player_id').textContent.trim();

        // Determine the corresponding arena div based on the player name
        if (playerId === opponent1Name) {
            return 'arena_left';
        } else if (playerId === opponent2Name) {
            return 'arena_center';
        } else if (playerId === opponent3Name) {
            return 'arena_right';
        } else if (playerId === playerName) {
            return 'arena_bottom';
        }
    },


    setStatus: function(playerId, status) {
        console.log("Status " + status + " of " + playerId);
        // Get the arena div
        const arenaDivId = this.findArena(playerId);
        const arenaDiv = document.getElementById(arenaDivId);
        // Display the message
        const passedElement = document.createElement('h1');
        passedElement.textContent = status;
        arenaDiv.innerHTML = '';
        arenaDiv.appendChild(passedElement);
    },

    setPlayedCard: function(playerId, cardIds) {
        const arenaDivId = this.findArena(playerId);

        console.log("Card(s) " + cardIds + " played by " + playerId);

        // Get the arena div
        const arenaDiv = document.getElementById(arenaDivId);
        // Clear any existing content in the arena div
        arenaDiv.innerHTML = '';

        // Check if the card_id array is empty
        if (!cardIds || cardIds.length === 0) {
            // Display the message
            const passedElement = document.createElement('h1');
            passedElement.textContent = "Passed";  // Display a meaningful message
            arenaDiv.appendChild(passedElement);
        } else {
            // Render the card(s)
            cardIds.forEach((card, i) => {
                let index = i - Math.floor(cardIds.length / 2);
                const cardElement = this.renderCard(card[0], card[1], index, false, false);
                arenaDiv.appendChild(cardElement);
            });
        }
    },

    playCards: function(cards) {
        console.log("playCards: " + cards);
        this.socket.emit('play_cards', {'cards': cards});
    },

    moveCardUp: function(...cardIds) {
        for (const cardId of cardIds) {
            const card = document.getElementById(cardId);
            if (card) {
                card.style.top = "0%";
                card.style.transform = "translateY(0%)";
                card.onclick = () => this.playCards(cardIds);
            }
        }
    },

    moveCardDown: function(...cardIds) {
        for (const cardId of cardIds) {
            const card = document.getElementById(cardId);
            if (card) {
                card.style.top = "100%";
                card.style.transform = "translateY(-100%)";
                card.onclick = undefined;
            }
        }
    },

    renderCard: function(value, suit, index, playerCard, playable) {
        const suitIndex = Object.keys(this.suits).indexOf(suit);

        let cardId;
        if (playerCard) {
            cardId = `${value}_${suitIndex}`;
        }
        else {
            // Non-player cards can't be accessed by mouse events, so call them something else
            cardId = `${value}__${suitIndex}`;
        }

        const cardHitArea = document.createElement("div");
        cardHitArea.className = "card_hit_area";
        cardHitArea.style.left = `${index+6}em`;
        cardHitArea.style.top = `${(index / 4) ** 2}em`;
        cardHitArea.style.transform = `rotate(${7 * index}deg)`;

        if (playable) {
            const playableCards = [cardId];
            for (let i = 0; i <= suitIndex; i++) {
                const similarCardId = `${value}_${i}`;
                const similarCard = document.getElementById(similarCardId);
                if (similarCard) {
                    playableCards.push(similarCardId);
                }
            }
            // Use this.moveCardUp and this.moveCardDown with proper binding
            cardHitArea.onmouseover = () => this.moveCardUp(...playableCards);
            cardHitArea.onmouseout = () => this.moveCardDown(...playableCards);
        }

        const card = document.createElement("div");
        card.className = playerCard ? "card" : "card_small";
        card.id = cardId;

        const front = document.createElement("div");
        front.className = suit === "♣" || suit === "♠" ? "front" : "front red";

        const indexTop = document.createElement("div");
        indexTop.className = "index";
        const indexBottom = document.createElement("div");
        indexBottom.className = "index_bottom";

        // Draw the Values to the side of the cards
        if ((value === 11) && (suit != "♠")) {
            // Special case for Ace
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
        if ((value >= 8 && value < 11) || (value === 13) || ((value === 11) && (suit == "♠"))) {
            const img = document.createElement("img");
            img.className = "face";
            img.width = 80;
            img.height = 130;
            const suitName = this.suits[suit];
            if (value === 8) img.src = `../static/img/jack_${this.suits[suit]}.jpg`;
            else if (value === 9) img.src = `../static/img/queen_${this.suits[suit]}.jpg`;
            else if (value === 10) img.src = `../static/img/king_${this.suits[suit]}.jpg`;
            else if (value === 13) img.src = `../static/img/${suit === "♠" ? "black_joker" : "red_joker"}.jpg`;
            else img.src = `../static/img/ace_spades.jpg`;
            front.appendChild(img);
        }

        card.appendChild(front);
        cardHitArea.appendChild(card);

        return cardHitArea;
    },

    populateAIDropdown: function(slotIndex, existingPlayers) {
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
        const availableAIs = this.aiOpponents.filter(ai =>
            !existingPlayers.includes(ai.split(" ")[0]));

        // Populate dropdown
        availableAIs.forEach(ai => {
            let option = document.createElement("option");
            option.value = ai.split(" ")[0];  // Extract the first name
            option.textContent = ai;
            aiDropdown.appendChild(option);
        });
    },

    updateOpponentSlot: function(index, playerName, cardCount, status, isOwner, existingPlayers) {
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
                let index = i - Math.floor(cardCount / 2);
                handElement.appendChild(this.renderCard(-1, "", index, false, false));
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
                this.populateAIDropdown(index, existingPlayers);
            }
        }
    },

    addAIPlayer: function(opponentIndex) {
        const aiDropdown = document.getElementById(`ai-opponent-${opponentIndex}`);
        const aiName = aiDropdown.value;

        // Find the full entry in AI_OPPONENTS array
        const aiEntry = this.aiOpponents.find(entry => entry.startsWith(aiName));

        // Extract difficulty from the parentheses (e.g., "Easy", "Medium", "Hard")
        const aiDifficulty = aiEntry.match(/\((.*?)\)/)[1];

        console.log(`Adding AI Player: ${aiName}, Difficulty: ${aiDifficulty}`);

        this.socket.emit("add_ai_player", { opponentIndex, aiName, aiDifficulty });
    },

    logOut: function() {
        console.log("Requesting logout");
        this.socket.emit('logout');
        // Also make HTTP request to ensure session is cleared
        fetch('/logout', {method: 'POST'})
            .then(() => window.location.href = '/');
    },

    updateState: function(content) {
        document.getElementById('debug_area').innerHTML = content;
    },

    leaveGame: function() {
        this.socket.emit('leave_game');
    },

    readyToStart: function() {
        console.log("readyToStart");
        this.socket.emit('start_game');
    }
};

// Initialize the game
document.addEventListener("DOMContentLoaded", function() {
    CardGame.init();

    // Expose functions needed by HTML event handlers
    window.playCards = (cards) => CardGame.playCards(cards);
    window.logOut = () => CardGame.logOut();
    window.leaveGame = () => CardGame.leaveGame();
    window.readyToStart = () => CardGame.readyToStart();
    window.addAIPlayer = (opponentIndex) => CardGame.addAIPlayer(opponentIndex);
});
