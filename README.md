# Asshole

This plays a game of Asshole with a console interface vs 3 computer opponents

Asshole is a card game where the player tries to get rid of all their cards

After the first episode, whoever finsished first is the King, follwed by Prince, Citizen and finally the looser asshole

Every subsequent episode the King swaps 2 shit cards for 2 good cards from the asshole

Prince swaps one shit card for a good card from the Citizen

Rules: 
* Typically played with a 54 card deck (will probably crash otherwise)
* Cards a ranked from 3 though to King, then Ace. Twos are ranked above Ace, and the two Jokers are the highest cards.

That is: 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2, Joker
* Asshole leads the first 'round' (or whoever has 3 Spades in the first episode where there are no positions)

Leader can play any card (but usally a low one)

*Leading a single Card*

If a single card is lead, other players must follow by a higher single card, or PASS.

After a PASS, that player must wait for the next 'round'

Play continues clockwise, with each player playing a higher card, or passing

Once everyone has passsed , the player of the highest card wins the 'round' and may now lead the next round

Players who passed may now play again

*Leading a double (or tripple, or 4 of a kind)*

If a double is lead, it can only be followed by a higher double. (Tripples only be higher tripples, 4 of a kind only by a higher 4 of a kind)
* Exception: A single 2 will beat any double. a double 2 will beat any tripple and tripple 2 will beat a 4 of a kind.
* Exception: A single Joker will beat any double or tripple, excpet tripple 2.
* Exception: A double Joker will beat any tripple, except tripple 2. It will beat a double 2.

In other words, a 2 can count as +1 cards when playing against multiple leads. The Joker counts as +2 cards. It follows that 4 twos in the highest play, since that would require 3 Jokers to beat. Double joker beats any other hand.


requires termcolor to draw colours otherwise the cards look like shit

python PlayAsshole.py

Offers 3 opponents of various cleverness
* PlayerSimple always plays the lowest possible card unless it's a set
* PlayerSplitter will split high cards to try to win
* PlayerHolder is actually pretty cunning and should not be underestimated

HumanPlayer prints to, and takes input from, the console

There are better UI implementations, but this is the minimum functionality to play a game 


The other interfaces are: 
* HTML Player is a HumanPlayer with an HTML5 interface
* Tensorflow player is intended to implement reinforcement learning]
* PyGamePlayer has a PyGame interface. duh.
