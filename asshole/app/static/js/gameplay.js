// Establish namespace to avoid global variables
const CardGame = {
    socket: null,
    aiOpponents: [
        "Alexi (Easy)", "Jordan (Easy)", "Kai (Easy)",
        "Samara (Medium)", "Quincy (Medium)", "Tian (Medium)",
        "Eshaan (Hard)", "Amari (Hard)", "Riven (Hard)"
    ],

    init: function() {
        this.socket = io.connect('http://' + document.domain + ':' + location.port);
        this.setupSocketEvents();
        this.cardRenderer = new CardRenderer();
        this.cardRenderer.setPlayFunction(this.playCards.bind(this));

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
            this.disableStartButton();
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_hand_start', () => this.socket.emit("request_game_state"));
        this.socket.on('notify_hand_won', (data) => {
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_played_out', (data) => {
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
            this.socket.emit("request_game_state");
        });
        this.socket.on('hand_won', (data) => {
            this.socket.emit("request_game_state");
        });
        this.socket.on('notify_player_turn', (data) => this.highlightCurrentPlayerTurn(data));
        this.socket.on('card_played', (data) => {
            // Just request the new state - setPlayedCard will be called by the game state receiver
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

    disableStartButton: function() {
        console.log("Disable start button");
        let startGameButton = document.getElementById("start_game_button");
        startGameButton.disabled = true;

        // Remove the animation class if it was added
        startGameButton.classList.remove("fill");

        // Clear any pending auto-click timeout
        if (this.startButtonTimeout) {
            clearTimeout(this.startButtonTimeout);
            this.startButtonTimeout = null;
        }
    },


    handleGameState: function(data) {
        console.log("Received full game state:", data);

        // Count how many opponents have joined (ignoring null slots)
        let playerCount = data.player_names.filter(opponent => opponent !== null).length;


        // Enable "Start Game" button if 3 opponents have joined
        let startGameButton = document.getElementById("start_game_button");
        if (playerCount >= 4 && startGameButton.disabled) {
            this.enableStartButton();
        }

        // Update opponent slots dynamically
        for (var i = 0; i < 3; i += 1) {
            this.updateOpponentSlot(i + 1, data.player_names[i+1], data.opponent_cards[i],
                                  data.is_owner, data.player_names);
        }

        // Update player's hand
        this.updatePlayerHand(data.player_hand);
        // Update player's name
        document.getElementById("player_id").innerHTML = data.player_names[0]

        const position_names = [
            "King",
            "Prince",
            "Citizen",
            "Asshole",
        ];

        for (var i = 0; i < 4; i += 1) {
            if (typeof data.player_status[i] === "string") {
                // The element for Absent players can't be found, so skip them
                if (data.player_status[i] !== "Absent") {
                    let position_index = data.player_positions.indexOf(data.player_names[i])
                    if (position_index >= 0) {
                        this.setStatus(data.player_names[i], "Finished as " + position_names[position_index] )
                    } else {
                        this.setStatus(data.player_names[i], data.player_status[i])
                    }
                }
            }
            else {
                // Some cards have been played
                this.setPlayedCard(data.player_names[i], data.player_status[i]);
            }
        }
    },

    updatePlayerHand: function(cards) {
        const handContainer = document.getElementById("player-hand");
        handContainer.innerHTML = ""; // Clear previous cards
        cards.forEach((card, i) => {
            let index = i - Math.floor(cards.length / 2);
            handContainer.appendChild(this.cardRenderer.renderCard(card[0], card[1], index, true, card[2]));
        });
    },


    highlightCurrentPlayerTurn: function(data) {
        const playerMap = {
            'opponent-1-name': 'playfield_left',
            'opponent-2-name': 'playfield_center',
            'opponent-3-name': 'playfield_right',
            'player_id': 'playfield_bottom'
        };

        Object.values(playerMap).forEach(id => {
            document.getElementById(id).classList.remove("active-turn");
        });

        Object.keys(playerMap).forEach(key => {
            if (document.getElementById(key).textContent.trim() === data.player) {
                document.getElementById(playerMap[key]).classList.add("active-turn");
            }
        });
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
                const cardElement = this.cardRenderer.renderCard(card[0], card[1], index, false, false);
                arenaDiv.appendChild(cardElement);
            });
        }
    },

    playCards: function(cards) {
        console.log("playCards: " + cards);
        this.socket.emit('play_cards', {'cards': cards});
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

    updateOpponentSlot: function(index, playerName, cardCount, isOwner, existingPlayers) {
        const nameElement = document.getElementById(`opponent-${index}-name`);
        const handElement = document.getElementById(`opponent-${index}-hand`);
        const aiSelectElement = document.getElementById(`opponent-${index}-ai-select`);

        if (playerName) {
            nameElement.textContent = playerName;
            handElement.innerHTML = ""; // Clear existing cards

            // Add cards if player has joined
            for (let i = 0; i < cardCount; i++) {
                let index = i - Math.floor(cardCount / 2);
                handElement.appendChild(this.cardRenderer.renderCard(-1, "", index, false, false));
            }

            // Hide AI selection (owner cannot add AI to an occupied slot)
            if (aiSelectElement) {
                aiSelectElement.style.display = "none";
            }
        } else {
            // Player has not joined
            nameElement.textContent = "Waiting for player to join...";
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
