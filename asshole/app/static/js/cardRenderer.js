class CardRenderer {
    constructor() {
        this.playCards = null;  // Initially null, will be set later
        this.suits = {
            "♠": "spades",
            "♣": "clubs",
            "♦": "diamonds",
            "♥": "hearts"
        };
    }

    setPlayFunction(playCardsFunction) {
        this.playCards = playCardsFunction;
    }


    renderCard(value, suit, index, playerCard, playable) {
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
    }

    moveCardUp(...cardIds) {
        for (const cardId of cardIds) {
            const card = document.getElementById(cardId);
            if (card) {
                card.style.top = "0%";
                card.style.transform = "translateY(0%)";
                card.onclick = () => this.playCards(cardIds);
            }
        }
    }

    moveCardDown(...cardIds) {
        for (const cardId of cardIds) {
            const card = document.getElementById(cardId);
            if (card) {
                card.style.top = "100%";
                card.style.transform = "translateY(-100%)";
                card.onclick = undefined;
            }
        }
    }
};

// Export as a module
window.CardRenderer = CardRenderer;
