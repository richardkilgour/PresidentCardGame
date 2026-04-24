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
        this.socket.on('notify_cards_swapped', (data) => {
            const myName = document.getElementById('player_id').textContent.trim();
            if (data.player_good === myName || data.player_bad === myName) {
                const iGave    = data.player_good === myName ? data.cards_to_bad  : data.cards_to_good;
                const iReceived = data.player_good === myName ? data.cards_to_good : data.cards_to_bad;
                const otherName = data.player_good === myName ? data.player_bad    : data.player_good;
                this.pendingSwap = { iGave, iReceived, otherName };
            }
        });
        this.socket.on('notify_hand_start', () => {
            if (this.pendingSwap) {
                this.showSwapModal(this.pendingSwap);
                this.pendingSwap = null;
            } else {
                this.socket.emit("request_game_state");
            }
        });
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

        // Auto-click after 5 seconds — store so disableStartButton can cancel it
        this.startButtonTimeout = setTimeout(() => {
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
        this.playerNames = data.player_names;
        for (var i = 0; i < 3; i += 1) {
            this.updateOpponentSlot(i + 1, data.player_names[i+1], data.opponent_cards[i],
                                  data.is_owner, data.player_names);
        }

        // Update player's hand
        this.updatePlayerHand(data.player_hand);
        this.updateStats(data.stats);
        const mustPass = data.is_my_turn
            && data.player_hand.length > 0
            && !data.player_hand.some(c => c[2]);
        const overlay = document.getElementById('compulsory-pass-overlay');
        overlay.style.display = mustPass ? 'flex' : 'none';
        // Update player's name
        document.getElementById("player_id").innerHTML = data.player_names[0]

        const position_names = [
            "President",
            "Vice-President",
            "Citizen",
            "Scumbag",
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
        const arenaDivId = this.findArena(playerId);
        const arenaDiv = document.getElementById(arenaDivId);
        delete arenaDiv.dataset.cardKey;
        arenaDiv.innerHTML = '';
        const passedElement = document.createElement('h1');
        passedElement.textContent = status;
        arenaDiv.appendChild(passedElement);
    },

    setPlayedCard: function(playerId, cardIds) {
        const arenaDivId = this.findArena(playerId);
        const arenaDiv = document.getElementById(arenaDivId);
        arenaDiv.innerHTML = '';

        if (!cardIds || cardIds.length === 0) {
            const passedElement = document.createElement('h1');
            passedElement.textContent = "Passed";
            arenaDiv.appendChild(passedElement);
            return;
        }

        // Skip re-render (and re-animation) if the same cards are already showing
        const cardKey = JSON.stringify(cardIds);
        if (arenaDiv.dataset.cardKey === cardKey) return;
        arenaDiv.dataset.cardKey = cardKey;

        const isMyArena = arenaDivId === 'arena_bottom';
        // Consume saved rects only for this play, then clear
        const srcRects = isMyArena ? this.lastPlayedRects : null;
        if (isMyArena) this.lastPlayedRects = null;

        // Average source position across all played cards (handles pairs/triples cleanly)
        let srcX = null, srcY = null;
        if (srcRects) {
            const valid = srcRects.filter(Boolean);
            if (valid.length) {
                srcX = valid.reduce((s, r) => s + r.left + r.width / 2, 0) / valid.length;
                srcY = valid.reduce((s, r) => s + r.top  + r.height / 2, 0) / valid.length;
            }
        }

        cardIds.forEach((card, i) => {
            const index = i - Math.floor(cardIds.length / 2);
            const cardElement = this.cardRenderer.renderCard(card[0], card[1], index, false, false);
            arenaDiv.appendChild(cardElement);

            const rotation = 7 * index;

            if (isMyArena) {
                // FLIP technique: measure destination, animate from hand position
                const dest = cardElement.getBoundingClientRect();
                const dx = srcX != null ? srcX - (dest.left + dest.width  / 2) : 0;
                const dy = srcY != null ? srcY - (dest.top  + dest.height / 2) : 180;
                cardElement.animate([
                    { transform: `translate(${dx}px, ${dy}px) rotate(0deg)`, opacity: '0.85' },
                    { transform: `rotate(${rotation}deg)`,                    opacity: '1'    }
                ], { duration: 420, easing: 'cubic-bezier(0.2, 0, 0.1, 1)' });
            } else {
                // Opponent card: flip from edge-on (face-down) to face-up
                const inner = cardElement.querySelector('.card_small');
                if (inner) {
                    inner.animate([
                        { transform: 'perspective(300px) rotateY(90deg)', opacity: '0.6' },
                        { transform: 'perspective(300px) rotateY(0deg)',   opacity: '1'   }
                    ], { duration: 380, easing: 'ease-out' });
                }
            }
        });
    },

    showSwapModal: function({ iGave, iReceived, otherName }) {
        const renderCards = (cards) => {
            const wrap = document.createElement('div');
            // card_hit_area uses position:absolute, so the container must be
            // position:relative with explicit dimensions to anchor them correctly.
            const cardWidth = 11; // em — left = (index+6)em, so 1 card needs ~11em
            wrap.style.cssText = `position:relative; height:150px; width:${cards.length * 2 + cardWidth}em; margin:0 auto;`;
            cards.forEach(([value, suit], i) => {
                const index = i - Math.floor(cards.length / 2);
                wrap.appendChild(this.cardRenderer.renderCard(value, suit, index, false, false));
            });
            return wrap;
        };

        const overlay = document.createElement('div');
        overlay.id = 'swap-modal-overlay';
        overlay.style.cssText = 'position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:1000; display:flex; align-items:center; justify-content:center;';

        const box = document.createElement('div');
        box.style.cssText = 'background:#222; color:#fff; border-radius:12px; padding:24px 32px; text-align:center; min-width:260px; max-width:480px;';

        const title = document.createElement('h2');
        title.textContent = 'Card Swap';
        title.style.marginTop = '0';
        box.appendChild(title);

        const giveLabel = document.createElement('p');
        giveLabel.textContent = `You give to ${otherName}:`;
        box.appendChild(giveLabel);
        box.appendChild(renderCards(iGave));

        const receiveLabel = document.createElement('p');
        receiveLabel.textContent = `You receive from ${otherName}:`;
        box.appendChild(receiveLabel);
        box.appendChild(renderCards(iReceived));

        const btn = document.createElement('button');
        btn.textContent = 'Got it';
        btn.style.cssText = 'margin-top:16px; padding:10px 28px; font-size:1.1em; cursor:pointer;';
        btn.addEventListener('click', () => this.dismissSwapModal());
        box.appendChild(btn);

        overlay.appendChild(box);
        document.body.appendChild(overlay);
    },

    dismissSwapModal: function() {
        const overlay = document.getElementById('swap-modal-overlay');
        if (overlay) overlay.remove();
        this.socket.emit("request_game_state");
    },

    updateStats: function(stats) {
        const panel = document.getElementById('stats-panel');
        if (!stats) { panel.innerHTML = ''; return; }

        const rows = stats.players.map(p => `
            <tr>
                <td>${p.current_position_icon ?? '—'}</td>
                <td>${p.name}</td>
                <td>${p.score >= 0 ? '+' : ''}${p.score}</td>
                <td>${p.max_consecutive_president > 0 ? '👑×' + p.max_consecutive_president : '—'}</td>
            </tr>`).join('');

        panel.innerHTML = `
            <table style="width:100%; border-collapse:collapse; margin-bottom:8px;">
                <thead>
                    <tr style="border-bottom:1px solid #888;">
                        <th></th>
                        <th style="text-align:left;">Player</th>
                        <th>Score</th>
                        <th>Best run</th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
            <div style="display:flex; justify-content:space-between; border-top:1px solid #888; padding-top:4px;">
                <span>Hands: ${stats.rounds_played}</span>
                <span>⬆ ${stats.high_score >= 0 ? '+' : ''}${stats.high_score}</span>
                <span>⬇ ${stats.low_score >= 0 ? '+' : ''}${stats.low_score}</span>
            </div>`;
    },

    playCards: function(cards) {
        console.log("playCards: " + cards);
        // Capture hand positions before the card leaves the DOM so we can animate from there
        this.lastPlayedRects = Array.isArray(cards)
            ? cards.map(id => { const el = document.getElementById(id); return el ? el.getBoundingClientRect() : null; })
            : null;
        this.socket.emit('play_cards', {'cards': cards});
    },


    populateAIDropdown: function(slotIndex, existingPlayers) {
        const aiDropdown = document.getElementById(`ai-opponent-${slotIndex}`);
        if (!aiDropdown) return;

        const savedValue = aiDropdown.value;
        aiDropdown.innerHTML = "";

        if (!Array.isArray(existingPlayers)) existingPlayers = [];

        // Collect names selected in the other visible dropdowns
        const otherSelected = [];
        for (let i = 1; i <= 3; i++) {
            if (i === slotIndex) continue;
            const other = document.getElementById(`ai-opponent-${i}`);
            if (other && other.style.display !== 'none' && other.value) {
                otherSelected.push(other.value);
            }
        }

        const taken = new Set([...existingPlayers.filter(Boolean), ...otherSelected]);
        const availableAIs = this.aiOpponents.filter(ai => !taken.has(ai.split(" ")[0]));

        availableAIs.forEach(ai => {
            let option = document.createElement("option");
            option.value = ai.split(" ")[0];
            option.textContent = ai;
            aiDropdown.appendChild(option);
        });

        // Restore previous selection if still available
        if (savedValue && [...aiDropdown.options].some(o => o.value === savedValue)) {
            aiDropdown.value = savedValue;
        }
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

                const select = document.getElementById(`ai-opponent-${index}`);
                if (select && !select._changeListenerAdded) {
                    select.addEventListener('change', () => {
                        for (let i = 1; i <= 3; i++) {
                            if (i !== index) this.populateAIDropdown(i, this.playerNames || existingPlayers);
                        }
                    });
                    select._changeListenerAdded = true;
                }
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
