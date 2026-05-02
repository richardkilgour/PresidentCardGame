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
        this.currentRoundRanks = {};   // player name → finish position (0=President…3=Scumbag)
        this.lastHandWinner   = null;  // winner of the most recent trick
        this.suppressPassOverlay = false;
        this.isAiControlled = false;
        this.currentTurnPlayer = null; // name of whoever's turn it currently is
        this.tutorialEnabled   = true; // centre hint text on/off
        this.hintsEnabled      = true; // mouseover tooltips on/off

        const startGameButton = document.getElementById("start_game_button");
        startGameButton.addEventListener("click", () => this.socket.emit("start_game"));
        startGameButton.disabled = true;

        this.initTooltip();
        this.setupPlayfieldTooltips();
    },

    // Debounced state request — collapses bursts of events into one round-trip
    requestGameState: function() {
        if (this._gsTimer) clearTimeout(this._gsTimer);
        this._gsTimer = setTimeout(() => {
            this._gsTimer = null;
            this.socket.emit("request_game_state");
        }, 30);
    },

    setupSocketEvents: function() {
        this.socket.on('connect', () => {
            console.log('WebSocket connection established');
            this.socket.emit('request_game_state');
        });
        this.socket.on("current_game_state", (data) => this.handleGameState(data));
        this.socket.on('notify_player_joined', () => this.requestGameState());
        this.socket.on('notify_game_started', () => {
            this.disableStartButton();
            this.requestGameState();
        });
        this.socket.on('notify_cards_swapped', (data) => {
            const myName = document.getElementById('player_id').textContent.trim();
            if (data.player_good === myName || data.player_bad === myName) {
                const iGave     = data.player_good === myName ? data.cards_to_bad  : data.cards_to_good;
                const iReceived = data.player_good === myName ? data.cards_to_good : data.cards_to_bad;
                const otherName = data.player_good === myName ? data.player_bad    : data.player_good;
                this.pendingSwap = { iGave, iReceived, otherName };
            }
        });
        this.socket.on('notify_hand_start', () => {
            this.currentRoundRanks = {};
            this.suppressPassOverlay = false;
            if (this.pendingSwap) {
                this.showSwapModal(this.pendingSwap);
                this.pendingSwap = null;
            } else {
                this.requestGameState();
            }
        });
        this.socket.on('notify_hand_won', (data) => {
            this.requestGameState();
        });
        this.socket.on('notify_played_out', (data) => {
            const rankSymbols = ["👑", "🥈", "🥉", "💩"];

            // Record this player's finish position for status display
            this.currentRoundRanks[data.player] = data.pos;

            // When the first player plays out (new President), move every still-playing
            // player's old rank badge to an ex-rank label
            if (data.pos === 0) {
                document.querySelectorAll("[data-rank]").forEach(el => {
                    el.setAttribute("data-ex-rank", el.getAttribute("data-rank"));
                    el.removeAttribute("data-rank");
                });
            }

            // Set the new rank badge for the player who just played out and strip any
            // ex-rank label they may have (they have a real rank now)
            const playerElement = this._findNameElement(data.player);
            if (playerElement) {
                playerElement.setAttribute("data-rank", rankSymbols[data.pos]);
                playerElement.removeAttribute("data-ex-rank");
            }

            // Once the Scumbag is determined all positions are known — ex-ranks can go
            if (data.pos === 3) {
                document.querySelectorAll("[data-ex-rank]").forEach(el => {
                    el.removeAttribute("data-ex-rank");
                });
            }

            this.requestGameState();
        });
        this.socket.on('hand_won', (data) => {
            this.lastHandWinner = data.winner;
            this.requestGameState();
        });
        this.socket.on('notify_player_turn', (data) => {
            this.highlightCurrentPlayerTurn(data);
        });
        this.socket.on('card_played', (data) => {
            const myName = document.getElementById('player_id').textContent.trim();
            if (data.player_id !== myName && data.card_id && data.card_id.length > 0) {
                this.suppressPassOverlay = false;
            }
            this.lastHandWinner = null;
            this.requestGameState();
        });

        // Disconnection / replacement events
        this.socket.on('player_disconnected', (data) => this.handlePlayerDisconnected(data.username));
        this.socket.on('replace_available', (data) => this.showReplaceButton(data.username));
        this.socket.on('player_replaced', (data) => {
            this.clearDisconnectedState(data.username);
            this.requestGameState();
        });
        this.socket.on('player_quit', (data) => {
            this.clearDisconnectedState(data.username);
            this.requestGameState();
        });
        this.socket.on('quit_confirmed', () => {
            window.location.href = '/';
        });
    },

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    _findNameElement: function(playerName) {
        for (const id of ['opponent-1-name', 'opponent-2-name', 'opponent-3-name', 'player_id']) {
            const el = document.getElementById(id);
            if (el && el.textContent.trim() === playerName) return el;
        }
        return null;
    },

    _playerPanels: function() {
        return [
            { nameId: 'player_id',       playfieldId: 'playfield_bottom' },
            { nameId: 'opponent-1-name', playfieldId: 'playfield_left'   },
            { nameId: 'opponent-2-name', playfieldId: 'playfield_center' },
            { nameId: 'opponent-3-name', playfieldId: 'playfield_right'  },
        ];
    },

    // -------------------------------------------------------------------------
    // Disconnect / replace UI
    // -------------------------------------------------------------------------

    handlePlayerDisconnected: function(username) {
        for (const { nameId, playfieldId } of this._playerPanels()) {
            const nameEl = document.getElementById(nameId);
            if (nameEl && nameEl.textContent.trim() === username) {
                const pf = document.getElementById(playfieldId);
                if (pf) pf.style.opacity = '0.4';
                break;
            }
        }
    },

    showReplaceButton: function(username) {
        const existing = document.getElementById('replace-btn');
        if (existing) existing.remove();

        const btn = document.createElement('button');
        btn.id = 'replace-btn';
        btn.dataset.username = username;
        btn.textContent = `Replace ${username} with AI`;
        btn.style.cssText = [
            'position:fixed', 'bottom:24px', 'left:50%',
            'transform:translateX(-50%)', 'padding:10px 24px',
            'font-size:1em', 'z-index:200', 'background:#c44',
            'color:#fff', 'border:none', 'border-radius:6px', 'cursor:pointer',
        ].join(';');
        btn.addEventListener('click', () => {
            this.socket.emit('replace_with_ai', { username });
            btn.remove();
        });
        document.body.appendChild(btn);
    },

    clearDisconnectedState: function(username) {
        const btn = document.getElementById('replace-btn');
        if (btn && btn.dataset.username === username) btn.remove();

        for (const { nameId, playfieldId } of this._playerPanels()) {
            const nameEl = document.getElementById(nameId);
            if (nameEl && nameEl.textContent.trim() === username) {
                const pf = document.getElementById(playfieldId);
                if (pf) pf.style.opacity = '';
                break;
            }
        }
    },

    // -------------------------------------------------------------------------
    // Game controls
    // -------------------------------------------------------------------------

    enableStartButton: function() {
        console.log("Enable start button");
        let startGameButton = document.getElementById("start_game_button");
        startGameButton.disabled = false;

        startGameButton.classList.add("fill");

        this.startButtonTimeout = setTimeout(() => {
            startGameButton.click();
        }, 5000);
    },

    disableStartButton: function() {
        console.log("Disable start button");
        let startGameButton = document.getElementById("start_game_button");
        startGameButton.disabled = true;

        startGameButton.classList.remove("fill");

        if (this.startButtonTimeout) {
            clearTimeout(this.startButtonTimeout);
            this.startButtonTimeout = null;
        }
    },


    // -------------------------------------------------------------------------
    // AI control toggle + speed
    // -------------------------------------------------------------------------

    toggleAiControl: function() {
        if (this.isAiControlled) {
            this.socket.emit('self_take_control');
        } else {
            this.socket.emit('self_play_as_ai');
        }
    },

    // val: 1–20 slider position → seconds interval (log scale)
    // val=1 → ~8s (very slow), val=10 → 1.0s (normal), val=20 → 0.1s (fast)
    _sliderToInterval: function(val) {
        return Math.pow(10, (10 - val) * 0.1);
    },

    _intervalToLabel: function(interval) {
        if (interval >= 3)   return 'Very slow';
        if (interval >= 1.5) return 'Slow';
        if (interval >= 0.8) return 'Normal';
        if (interval >= 0.3) return 'Fast';
        return 'Very fast';
    },

    onSpeedSlider: function(val) {
        const interval = this._sliderToInterval(parseInt(val, 10));
        document.getElementById('speed-label-value').textContent = this._intervalToLabel(interval);
        this.socket.emit('set_game_speed', { interval });
    },

    // Sync slider position from a server-provided interval value
    _syncSpeedSlider: function(interval) {
        // Invert: val = 10 - log10(interval) / 0.1
        const val = Math.round(10 - Math.log10(interval) / 0.1);
        const clamped = Math.max(1, Math.min(20, val));
        const slider = document.getElementById('speed-slider');
        if (slider && parseInt(slider.value, 10) !== clamped) {
            slider.value = clamped;
            document.getElementById('speed-label-value').textContent = this._intervalToLabel(interval);
        }
    },

    handleGameState: function(data) {
        console.log("Received full game state:", data);

        const position_names = [
            "President",
            "Vice-President",
            "Citizen",
            "Scumbag",
        ];

        let playerCount = data.player_names.filter(opponent => opponent !== null).length;

        let startGameButton = document.getElementById("start_game_button");
        if (playerCount >= 4 && startGameButton.disabled) {
            this.enableStartButton();
        }

        this.playerNames = data.player_names;
        for (var i = 0; i < 3; i += 1) {
            this.updateOpponentSlot(i + 1, data.player_names[i+1], data.opponent_cards[i],
                                  data.is_owner, data.player_names);
        }

        this.updatePlayerHand(data.player_hand);
        this.updateStats(data.stats);
        this.updatePlayerAvatar(data.player_names[0]);
        this.updateTutorialText(data);

        // Sync AI-control button label
        this.isAiControlled = !!data.is_ai_controlled;
        const aiBtn = document.getElementById('ai-control-btn');
        if (aiBtn) aiBtn.textContent = this.isAiControlled ? 'Take Control' : 'Play as AI';

        // Sync speed slider without triggering another emit
        if (data.step_interval !== undefined) this._syncSpeedSlider(data.step_interval);

        const mustPass = !this.suppressPassOverlay
            && data.is_my_turn
            && data.player_hand.length > 0
            && !data.player_hand.some(c => c[2]);
        const overlay = document.getElementById('compulsory-pass-overlay');
        overlay.style.display = mustPass ? 'flex' : 'none';
        document.getElementById("player_id").innerHTML = data.player_names[0];

        for (var i = 0; i < 4; i += 1) {
            if (typeof data.player_status[i] === "string") {
                if (data.player_status[i] !== "Absent") {
                    const rankIndex = this.currentRoundRanks[data.player_names[i]];
                    if (rankIndex !== undefined) {
                        this.setStatus(data.player_names[i], "Finished as " + position_names[rankIndex]);
                    } else if (data.player_status[i] === 'Waiting' && data.player_names[i] === this.lastHandWinner) {
                        this.setStatus(data.player_names[i], 'Winner!');
                    } else {
                        this.setStatus(data.player_names[i], data.player_status[i]);
                    }
                }
            }
            else {
                this.setPlayedCard(data.player_names[i], data.player_status[i]);
            }
        }
    },

    updatePlayerHand: function(cards) {
        const handContainer = document.getElementById("player-hand");
        handContainer.innerHTML = "";
        cards.forEach((card, i) => {
            let index = i - Math.floor(cards.length / 2);
            handContainer.appendChild(this.cardRenderer.renderCard(card[0], card[1], index, true, card[2]));
        });
        // Add tooltip hints to each card hit area
        handContainer.querySelectorAll('.card_hit_area').forEach((hitArea, i) => {
            if (!cards[i]) return;
            const tip = cards[i][2]
                ? 'Hover to lift · click to play this card'
                : 'Cannot play — rank or count doesn\'t match the current lead';
            hitArea.addEventListener('mouseenter', () => this.showTooltip(tip));
            hitArea.addEventListener('mouseleave', () => this.hideTooltip());
        });
    },


    highlightCurrentPlayerTurn: function(data) {
        this.currentTurnPlayer = data.player; // track for tutorial text

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
        this.requestGameState();
    },

    findArena: function(playerId) {
        const opponent1Name = document.getElementById('opponent-1-name').textContent.trim();
        const opponent2Name = document.getElementById('opponent-2-name').textContent.trim();
        const opponent3Name = document.getElementById('opponent-3-name').textContent.trim();
        const playerName = document.getElementById('player_id').textContent.trim();

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

    // Centre a group of .card_small elements horizontally inside arenaDiv.
    // Must be called after all cards are appended but before animations fire.
    _centerMeldSpread: function(arenaDiv) {
        const cards = Array.from(arenaDiv.querySelectorAll('.card_small'));
        if (cards.length === 0) return;
        // card_small font-size is 15pt = 20 px; getComputedStyle for safety
        const emPx  = parseFloat(getComputedStyle(cards[0]).fontSize);
        const contW = arenaDiv.offsetWidth;          // px width of the meld box
        const n     = cards.length;
        // Spread width = (n-1) inter-card gaps of 1 em each + one card width (3.75 em)
        const spreadW = ((n - 1) + 3.75) * emPx;
        // Left pixel that centres the spread
        const startPx = (contW - spreadW) / 2;
        cards.forEach((card, i) => {
            card.style.left = (startPx / emPx + i) + 'em';
        });
    },

    setPlayedCard: function(playerId, cardIds) {
        const arenaDivId = this.findArena(playerId);
        const arenaDiv = document.getElementById(arenaDivId);

        const cardKey = JSON.stringify(cardIds);
        if (arenaDiv.dataset.cardKey === cardKey) return;
        arenaDiv.dataset.cardKey = cardKey;

        arenaDiv.innerHTML = '';

        if (!cardIds || cardIds.length === 0) {
            const passedElement = document.createElement('h1');
            passedElement.textContent = "Passed";
            arenaDiv.appendChild(passedElement);
            return;
        }

        const isMyArena = arenaDivId === 'arena_bottom';
        const srcRects = isMyArena ? this.lastPlayedRects : null;
        if (isMyArena) this.lastPlayedRects = null;

        let srcX = null, srcY = null;
        if (srcRects) {
            const valid = srcRects.filter(Boolean);
            if (valid.length) {
                srcX = valid.reduce((s, r) => s + r.left + r.width / 2, 0) / valid.length;
                srcY = valid.reduce((s, r) => s + r.top  + r.height / 2, 0) / valid.length;
            }
        }

        // Pass 1 — render all cards into the DOM
        const rendered = cardIds.map((card, i) => {
            const index = i - Math.floor(cardIds.length / 2);
            const cardElement = this.cardRenderer.renderCard(card[0], card[1], index, false, false);
            arenaDiv.appendChild(cardElement);
            return { el: cardElement, rotation: 7 * index };
        });

        // Pass 2 — centre the spread so COG sits at the container's midpoint
        this._centerMeldSpread(arenaDiv);

        // Pass 3 — animate (uses final centred positions for rect calculations)
        rendered.forEach(({ el, rotation }) => {
            if (isMyArena) {
                const dest = el.getBoundingClientRect();
                const dx = srcX != null ? srcX - (dest.left + dest.width  / 2) : 0;
                const dy = srcY != null ? srcY - (dest.top  + dest.height / 2) : 180;
                el.animate([
                    { transform: `translate(${dx}px, ${dy}px) rotate(0deg)`, opacity: '0.85' },
                    { transform: `rotate(${rotation}deg)`,                    opacity: '1'    }
                ], { duration: 420, easing: 'cubic-bezier(0.2, 0, 0.1, 1)' });
            } else {
                el.animate([
                    { transform: 'perspective(300px) rotateY(90deg)', opacity: '0.6' },
                    { transform: 'perspective(300px) rotateY(0deg)',   opacity: '1'   }
                ], { duration: 380, easing: 'ease-out' });
            }
        });
    },

    showSwapModal: function({ iGave, iReceived, otherName }) {
        const renderCards = (cards) => {
            const wrap = document.createElement('div');
            // Use fixed px so card_hit_area's 20pt font-size doesn't cause em-unit mismatch
            wrap.style.cssText = 'position:relative; height:185px; width:420px; margin:0 auto;';
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
        box.style.cssText = 'background:#222; color:#fff; border-radius:12px; padding:24px 32px; text-align:center; min-width:260px; max-width:520px;';

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
        this.requestGameState();
    },

    updateSeatAvatar: function(index, playerName) {
        const el = document.getElementById(`opponent-${index}-avatar`);
        if (!el) return;
        el.innerHTML = '';
        el.className = 'seat-avatar';
        if (!playerName) return;
        const isAI = playerName.endsWith(' (AI)');
        if (isAI && typeof window.getAvatarSVG === 'function') {
            el.innerHTML = window.getAvatarSVG(playerName, '#00FFB8', 32);
        } else {
            el.classList.add('human-avatar');
            el.textContent = playerName.slice(0, 2).toUpperCase();
        }
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

    // -------------------------------------------------------------------------
    // Player avatar (sidebar)
    // -------------------------------------------------------------------------

    updatePlayerAvatar: function(playerName) {
        const el = document.getElementById('player-avatar');
        if (!el) return;
        el.innerHTML = '';
        el.className = 'seat-avatar';
        if (!playerName) return;
        const isAI = playerName.endsWith(' (AI)');
        if (isAI && typeof window.getAvatarSVG === 'function') {
            el.innerHTML = window.getAvatarSVG(playerName, '#00FFB8', 28);
        } else {
            el.classList.add('human-avatar');
            el.textContent = playerName.slice(0, 2).toUpperCase();
        }
    },

    // -------------------------------------------------------------------------
    // Centre tutorial / hint text
    // -------------------------------------------------------------------------

    updateTutorialText: function(data) {
        const el     = document.getElementById('table-center-info');
        const msgEl  = document.getElementById('center-msg');
        if (!el || !msgEl) return;

        if (!this.tutorialEnabled) {
            el.style.display = 'none';
            return;
        }
        el.style.display = 'flex';

        let msg = '';

        if (!data.player_hand || data.player_hand.length === 0) {
            // Game hasn't started yet
            const joined = (data.player_names || []).filter(Boolean).length;
            msg = joined < 4 ? `Waiting for players… (${joined}/4)` : 'Ready — press Start Game';
        } else if (data.is_my_turn) {
            const playable = data.player_hand.filter(c => c[2]).length;
            msg = playable > 0
                ? '▲ Your turn — hover a card and click to play'
                : '▲ Your turn — no playable cards, you must pass';
        } else {
            msg = this.currentTurnPlayer ? `${this.currentTurnPlayer}'s turn` : '';
        }

        msgEl.textContent = msg;
    },

    // -------------------------------------------------------------------------
    // Floating tooltip
    // -------------------------------------------------------------------------

    initTooltip: function() {
        this._tooltipEl = document.getElementById('game-tooltip');
        document.addEventListener('mousemove', (e) => {
            if (this._tooltipEl && this._tooltipEl.style.display !== 'none') {
                // Keep tooltip 14px right/below cursor; flip left if near right edge
                const gap  = 14;
                const tw   = this._tooltipEl.offsetWidth;
                const left = (e.clientX + gap + tw > window.innerWidth)
                           ? e.clientX - tw - gap
                           : e.clientX + gap;
                this._tooltipEl.style.left = left + 'px';
                this._tooltipEl.style.top  = (e.clientY + gap) + 'px';
            }
        });
    },

    showTooltip: function(text) {
        if (!this.hintsEnabled || !this._tooltipEl || !text) return;
        this._tooltipEl.textContent = text;
        this._tooltipEl.style.display = 'block';
    },

    hideTooltip: function() {
        if (this._tooltipEl) this._tooltipEl.style.display = 'none';
    },

    // Attach tooltip behaviour to any element carrying data-tooltip attribute.
    // Called once on init; also handles dynamically-added elements via delegation.
    setupPlayfieldTooltips: function() {
        document.addEventListener('mouseover', (e) => {
            const target = e.target.closest('[data-tooltip]');
            if (target) this.showTooltip(target.getAttribute('data-tooltip'));
        });
        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('[data-tooltip]')) this.hideTooltip();
        });
    },

    playCards: function(cards) {
        console.log("playCards: " + cards);
        this.suppressPassOverlay = true;
        this.lastHandWinner = null;
        this.lastPlayedRects = Array.isArray(cards)
            ? cards.map(id => { const el = document.getElementById(id); return el ? el.getBoundingClientRect() : null; })
            : null;
        // Optimistically remove played cards from hand so the UI responds immediately
        if (Array.isArray(cards)) {
            cards.forEach(id => {
                const el = document.getElementById(id);
                if (el) el.closest('.card_hit_area')?.remove();
            });
        }
        this.socket.emit('play_cards', {'cards': cards});
    },


    populateAIDropdown: function(slotIndex, existingPlayers) {
        const aiDropdown = document.getElementById(`ai-opponent-${slotIndex}`);
        if (!aiDropdown) return;

        const savedValue = aiDropdown.value;
        aiDropdown.innerHTML = "";

        if (!Array.isArray(existingPlayers)) existingPlayers = [];

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
            handElement.innerHTML = "";

            for (let i = 0; i < cardCount; i++) {
                let index = i - Math.floor(cardCount / 2);
                handElement.appendChild(this.cardRenderer.renderCard(-1, "", index, false, false));
            }

            this.updateSeatAvatar(index, playerName);

            if (aiSelectElement) {
                aiSelectElement.style.display = "none";
            }
        } else {
            nameElement.textContent = "Waiting for player to join...";
            this.updateSeatAvatar(index, null);
            handElement.innerHTML = "";
            console.log("isOwner = " + isOwner);
            console.log("aiSelectElement = " + aiSelectElement);
            if (isOwner && aiSelectElement) {
                aiSelectElement.style.display = "flex";
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

        const aiEntry = this.aiOpponents.find(entry => entry.startsWith(aiName));
        const aiDifficulty = aiEntry.match(/\((.*?)\)/)[1];

        console.log(`Adding AI Player: ${aiName}, Difficulty: ${aiDifficulty}`);

        this.socket.emit("add_ai_player", { opponentIndex, aiName, aiDifficulty });
    },

    logOut: function() {
        console.log("Requesting logout");
        this.socket.emit('logout');
        fetch('/logout', {method: 'POST'})
            .then(() => window.location.href = '/');
    },

    updateState: function(content) {
        document.getElementById('debug_area').innerHTML = content;
    },

    leaveGame: function() {
        if (confirm('Quit this game? You will be replaced by an AI player and cannot rejoin.')) {
            this.socket.emit('quit_game');
        }
    },

    readyToStart: function() {
        console.log("readyToStart");
        this.socket.emit('start_game');
    }
};

// Initialize the game
document.addEventListener("DOMContentLoaded", function() {
    CardGame.init();

    window.playCards       = (cards) => CardGame.playCards(cards);
    window.logOut          = ()      => CardGame.logOut();
    window.leaveGame       = ()      => CardGame.leaveGame();
    window.readyToStart    = ()      => CardGame.readyToStart();
    window.addAIPlayer     = (i)     => CardGame.addAIPlayer(i);
    window.toggleAiControl = ()      => CardGame.toggleAiControl();
    window.onSpeedSlider   = (val)   => CardGame.onSpeedSlider(val);

    // Toggle functions wired to the checkboxes in the controls panel
    window.toggleTutorial = (enabled) => {
        CardGame.tutorialEnabled = enabled;
        const el = document.getElementById('table-center-info');
        if (el) el.style.display = enabled ? 'flex' : 'none';
    };
    window.toggleTooltips = (enabled) => {
        CardGame.hintsEnabled = enabled;
        if (!enabled) CardGame.hideTooltip();
    };
});
