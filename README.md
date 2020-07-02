# Tennis_Network_Analysis
Created a weighted and directed network using the ATP 2019 tour. The nodes represent the top 300 players, and the edges exist if a player wins against another player.
The tennis network is a weighted and directed network with the players representing and nodes and an edge exists if a player has a win against another player. The weights are counted as the number of wins a player has against a particular opponent. I scraped the data from the website http://www.tennisabstract.com/. The number of nodes(tennis players) is 384.
I had to separate all the information on the website to particularly get the Male players who have played in ATP tournaments in the year 2019. I used various centrality measures to rank the players based on win/loss percentage. I applied community detection to the network to see different group of players with high interconnectivity and see players who are of
similar level. I used Cocitation graph for community detection.

From analyzing this network, I observe that some players who are ranked at the bottom of the table play very few to no matches than others who top the leaderboard. They will have to play against players of similar rankings and rank up before they are able to play against the higher ranked players.
