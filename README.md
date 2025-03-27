# President

This plays a game of President with a console interface vs 3 computer opponents

President is a card game where the player tries to get rid of all their cards

After the first episode, whoever finished first is the President, followed by Vice-President, Citizen and finally the looser, scumbag

Every subsequent episode the President swaps 2 bad cards for 2 good cards from the scumbag

Vice-President swaps one bad card for a good card from the Citizen

Rules: 
* Typically played with a 54 card deck (will probably crash otherwise)
* Cards a ranked from 3 though to King, then Ace. Twos are ranked above Ace, and the two Jokers are the highest cards.

That is: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, Joker
* Scumbag leads the first 'round' (or whoever has 3 Spades in the first episode where there are no positions)

Leader can play any card (but usally a low one)

*Leading a single Card*

If a single card is lead, other players must follow by a higher single card, or PASS.

After a PASS, that player must wait for the next 'round'

Play continues clockwise, with each player playing a higher card, or passing

Once everyone has passed, the remaining player (with the highest card) wins the 'round' and may now lead the next round

Players who passed may now play again

*Leading a double (or triple, or 4 of a kind)*

If a double is lead, it can only be followed by a higher double; Triples only by higher triples; 4 of a kind only by a higher 4 of a kind)
* Exception: A single 2 will beat any double; A double 2 will beat any triple; A triple 2 will beat a 4 of a kind.
* Exception: A single Joker will beat any double or triple, except triple 2.
* Exception: A double Joker will beat any triple, except triple 2; It will beat a double 2.

In other words, a 2 can count as +1 cards when playing against multiple leads. The Joker counts as +2 cards. It follows that 4 twos in the highest play, since that would require 3 Jokers to beat. Double joker beats any other hand.

## User Interface

There is implementations to:
* Play on the console (requires termcolor to draw colours otherwise the cards look weird) HumanPlayer prints to, and takes input from, the console. Usage: python PlayPresident.py 
* PLay as HTML using flask (see app directory)
* PyGame (ui directory - under development)