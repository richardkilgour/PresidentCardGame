# Asshole

Play a game of Asshole on the console

requires termcolor to draw colours otherwise the cards look like shit

python PlayAsshole.py

Offers 3 opponents of various cleverness
PlayerSimple always plays the lowest possible card unless it's a set
PlayerSplitter will split high cards to try to win
PlayerHolder is actually pretty cunning and should not be underestimated

HumanPlayer prints to, and takes input from, the console

There are better UI implementations, but this is the minimum functionality to play a game 


The other interfaces are: 
* HTML Player is a HumanPlayer with an HTML5 interface
* Tensorflow player is intended to implement reinforcement learning]
* PyGamePlayer has a PyGame interface. duh.