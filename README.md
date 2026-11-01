So I made a game as you can see... and got a little carried away... 

I've given the rulebook, some algorithms I used to generate games, and a document outlining my findings!

Feel free to do your own analysis, or just have fun with the game! (Also I'd love feedback if you have any :-) )

Anyway, the plan for future analysis is to optimize the algorithm to calculate your chance of winning differently. Currently, it runs up to 2000 different combinations of moves - it quite literally tries looking at every outcome. This is bad. At least for time and computation. I'd like to see if as I run each possible play, I track how many points I win / lose by. If I happen to see a normal distribution, which I'd hope to expect, I'll make an algorithm which runs 20 possibilities, and continues testing more possibilities until both the standard deviation and mean approach a value within a certain tolerance - once met, this is the predicted normal distribution for that play and I don't need to run a billion more possibilities.
