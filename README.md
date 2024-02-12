# CSE 151A Milestone 2: Data Exploration and Initial Preprocessing

The following README.md primarily provides data interpretation and supplementary / extra details about the data visualizations and calculations that are done in the notebook. Please also look into the notebook for more information. 

Group Member List: 

- Aaron Li all042\@ucsd.edu
- Bryant Jin brjin\@ucsd.edu
- Daniel Ji daji\@ucsd.edu
- David Wang dyw001\@ucsd.edu
- Dhruv Kanetkar dkanetkar\@ucsd.edu
- Eric Ye e2ye\@ucsd.edu
- Kevin Lu k8lu\@ucsd.edu
- Kevin Shen k3shen\@ucsd.edu
- Max Weng maweng\@ucsd.edu
- Roshan Sood rosood\@ucsd.edu

Abstract, for reference: 

Although sports analytics captured national attention only in 2011 with the release of Moneyball, research in the field is nearly a century old. Until relatively recently, this research was largely done by hand; however, the heavily quantitative nature of sports analytics makes it an attractive target for machine learning. This paper explores the application of advanced machine learning models to predict team performance in National Basketball Association (NBA) regular season and playoff games. Several models were trained on a rich dataset spanning 73 years, which includes individual player metrics, opponent-based performance, and team composition. The core of our analysis lies in combining individual player metrics, opponent-based game performances, and team chemistry, extending beyond traditional stat line analysis by incorporating nuanced factors. We employ various machine learning techniques, including neural networks and gradient boosting machines, to generate predictive models for player performance and compare their performance with both each other and traditional predictive models. Our analysis suggests that gradient boosting machines and neural networks significantly outperform other models. Neural networks demonstrate significant effectiveness in handling complex, non-linear data interrelations, while gradient boosting machines excel in identifying intricate predictor interactions. Our findings emphasize the immense potential of sophisticated machine learning techniques in sports analytics and mark a growing shift towards computer-aided and computer-based approaches in sports analytics.


## Data Exploration

Preface: In our meetings these past weeks, we finalized our plan with the dataset and determined more of the model-specific details about the task we outlined in our abstract. Because we wanted to explore team performance (in the regular season for now as a MVP, we can extend our output features / variables we want to predict to possibly the playoffs later on), we decided that the best metric to measure this was the wins and losses of the team in the regular season, and to normalize it, we’d just get the win percentage. **Note** that we are looking to **predict team performance of any arbitrary team of 10 players**, not just pre-existing teams. ****With this in mind, we began our data exploration for potential interesting player performance, player experience, team performance, and opponent team performance features that would be good to try to build a machine learning model with. 


### Data Interpretation (What the Data Means)

The Kaggle Dataset we are using is a **very comprehensive dataset** with many features, spanning from individual player stats to team performance measurements for nearly the entirety of the NBA. Here is a breakdown of some of the features we found interesting while exploring our data:


#### Across all CSVs

All-season related data files contain season identification (what season, what type of league), all team-related data files contain team identification (team name, abbreviation), and all player-related data files contain player identification and other information (name, age, position, etc.). Because players may have the same name, we can use the player\_id column to distinguish.\
**Regarding team data files**, although we are predicting team performance based on the players on that team, this team data would be possibly useful in scaling the players performance accordingly depending on their team. For example, a player may not necessarily have a good plus/minus, but that might be because their team overall is not a good team and the overall negative team performance affects their own performance. Nevertheless, we still focused on taking a deeper dive into player stats more than team stats when exploring data. 


#### Advanced.csv

Some of the more complex measurements for a player’s performance. Using these stats, which are feature-engineered from more simple stats (points, rebounds, assists, minutes played, etc.), we can get a more holistic measurement of individuals. Interesting columns: 

1. **per**: Player efficiency rating, a measure of per-minute production **standardized** such that the league average is 15. 

2. **usg\_percent**: Usage Percentage, an estimate of the percentage of team plays used by a player while they were on the floor. **Normalized** from 0 to 1 since it’s a percentage. 

3. **ws** and **ws\_48**: Win Shares, an estimate of the number of wins contributed by a player. Also exists WS/48 which is win shares per 48 minutes, a **scaled** feature. 

4. **bpm**: Box Plus/Minus, a box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team (**scaled**).

5. **vorp**: Value over Replacement Player, a box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.


#### All-Star Selections.csv

Lists all-star selections, voted by current NBA players and fans. After reviewing all data features from this dataset, although award selection likely has a strong correlation with player performance, we decided to look more into the numerical objective statistics and exclude awards and subjective data features (based on fellow player / media perception) like this data file from our project (all these files would make our project scope too large, this dataset is huge!).


#### End of Season Teams.csv, End of Season Teams (Voting).csv

Lists the best players, in their respective positions (30 players total by season), voted by the media. For similar reasons to All-Star voting, we did not decide to use this data. 


#### Per 100 Poss.csv, Per 36 Minutes.csv, Player Totals, Player Per Game.csv

These datafiles hold the same columns, with slight differences but with Per 100 Poss.csv being the most fit since it is both normalized to the other stats (Per 100 Poss. for teams and opposing teams) and contains the most information (also has offensive and defensive ratings of players). Here are the key features of interest, both raw and slightly feature-engineered: 

1. **fga** and **fg\_percent**: field goals (total attempts) and their accuracy (how many they made / how many they attempted). Excluded field goals because it can be derived from the two easily (will do so for all metrics like this). 

2. **x2pa** and **x2p\_percent**, **x3pa** and **x3p\_percent**: same as fga and fga\_percent, but instead just two / three pointers 

3. **e\_fg\_percent**: effective field goal percentage 

4. **trb** (total rebound percentage)

5. and raw stats: **ast** (assists), **stl** (steals), **blk** (blocks), **tov** (turnovers), **pts** (points)


#### Player Award Shares.csv

Player awards (like end of season teams voting, all-star selections, MVP, rookie of the year, defensive player of the year, most improved player, etc.). Decided not to use after reviewing all files, see All-Star Selections.csv for elaboration.


#### Player Career Info.csv

Lists the first, last, and number of seasons that an NBA player was active. Note that if a NBA player was not playing in the NBA for some seasons, but returned later, the number of seasons may be less than the number of seasons between the first and last season the NBA player was active. 


#### Player Season Info.csv

List of NBA players and age, birth year, team, position, and experience (how many years they’ve played) for every season. Like Player Career Info.csv, is helpful supplemental information that can be possible features for calculating team composition / chemistry (how long teammates have stuck together). The two features can possibly used in feature engineering to develop a score of how long a team has stuck together and how much players on the team have played as teammates. 


#### Player Play by Play.csv

List of player stats, related to the player’s actions / performance based on possessions (plays).

1. **g** and **mp**: games and minutes played, useful for potential standardization of these metrics. 

2. **on\_court\_plus\_minus\_per\_100\_poss**, **net\_plus\_minus\_per\_100\_poss**: metric to measure change score (own team scored points - opponent team scored points) while each player is on the court (and also off the court for net plus/minus). Useful for measuring holistic player performance (not just raw features like points, rebounds, assists, etc. stats)

3. **points\_generated\_by\_assists**: points generated by assist, all points for which the player gets an assist for


#### Player Shooting.csv

Detailed player shooting statistics. Many of these columns are definitely too specific for predicting team performance from player stats and wouldn’t be used, but some generalized ones could be interesting to model with:

1. **fg\_percent**: a raw feature of the field goal percentage of a player

2. **avg\_dist\_fga**: the average distance of a field goal attempt - a combination of far and close shooters (guards and conventional forwards/centers) or just far shooters (guards and forwards/centers that can also shoot from far away OR small-ball team with guards) could possibly result in a better team than other combinations


#### Team Stats Per 100 Poss.csv, Team Stats Per Game.csv, Team Totals.csv

Also includes many key features (nearly all raw features) about teams. The raw features are as follows (per 100 possessions, as named; stats per game and totals are excluded because they are more or less the same with just different values)

1. **fga\_per\_100\_poss** and **fg\_percent** (see above)

2. **x3pa\_per\_100\_poss** and **x3p\_percent** (see above)

3. **x2pa\_per\_100\_poss** and **x2p\_percent** (see above)

4. **trb\_per\_100\_poss** (true rebounding) 

5. **ast\_per\_100\_poss** (assists)

6. **stl\_per\_100\_poss** (steals)

7. **blk\_per\_100\_poss** (blocks)

8. **tov\_per\_100\_poss** (turnovers)

9. **pts\_per\_100\_poss** (points)


#### Team Summaries.csv

Includes both raw and statistically calculated metrics about teams, where the statistically calculated metrics may give a better measure of the team than actual results, since they attempt to reduce luck and other random factors. 

1. **w** and **l**, **pw** and **pl**:

2. **mov**: margin of victory,

3. **o\_rtg, d\_rtg, n\_rtg**: net rating,

4. **e\_fg\_perrcent** and **opp\_e\_fg\_percent**: effective field goal percentage  


#### Opponent Stats Per 100 Pos.csv, Opponent Stats Per Game.csv, Opponent Totals.csv

1. **opp\_fga\_per\_100\_poss** and **fg\_percent** (see above)

2. **opp\_x3pa\_per\_100\_poss** and **x3p\_percent** (see above)

3. **opp\_x2pa\_per\_100\_poss** and **x2p\_percent** (see above)

4. **opp\_trb\_per\_100\_poss** (true rebounding) 

5. **opp\_ast\_per\_100\_poss** (assists)

6. **opp\_stl\_per\_100\_poss** (steals)

7. **opp\_blk\_per\_100\_poss** (blocks)

8. **opp\_tov\_per\_100\_poss** (turnovers)

9. **opp\_pts\_per\_100\_poss** (points)


#### Summary

Here are the files we are potentially interested in for modeling: 

1. Advanced.csv

2. Player Play by Play.csv

3. Player per Game.csv

4. Per 100 Pos.csv + Player Shooting.csv (can be merged in one)

5. Team Stats per 100 Poss.csv

6. Team Summaries.csv

7. Opponents Stats Per 100 Poss.csv

The notebook in this Github repository explore these 7 data files.


### Visualize The Data

Please see the notebook for visualizations.


## Data Preprocessing and Calculation

### Missing Data

Regarding missing columns or data, **given that we are looking into NBA data after 1990**, the data files were well filled out and comprehensive (please see below for pre-1990 NBA data). Please read more below regarding the entire dataset (since the start of the NBA). Because there are so many players in the NBA and it would be quite hard to check for missing rows, we generally assumed that the \~11,000 filtered NBA player data observations (post-1990) were accurate. We assume likewise that NBA team data post-1990 was accurate and complete as well.  With columns, we checked for NaN values and dropped rows that had them (there were very few instances, if not none, for our data files, so this wouldn’t skew our data in any way). 

For example, in the player Advanced.csv, we ran: 

```
print(advanced_player_df_stats.isna().values.any())
print(advanced_player_df_stats.isnull().values.any())
```
Where `advanced_player_df_stats` contained `'decade', 'experience', 'per', 'usg_percent', 'ws_48', 'bpm', 'vorp'`. The output for both lines was `false`, indicating we likely had no missing data. 

Looking at the dataset as a whole, imputations came from many different fields from our \~35,000+ player data entries being filled with NaN values. This is because in our combined database, much of the past seasons did not record certain attributes that we were intending to use for our machine learning. Thus, after filtering out a bit of the NaN's we found that we can discard seasons 1947-1973, as they did not record any useful data such as assist percentage, steal percentage, etc. Thus, shortening the dataset around then still kept us with around \~25,000+ player data entries, which would be sufficient for our machine learning algorithm. Some attributes were only being recorded starting very recently, which led us to removing these options completely. For example, Games Started was only recorded starting 1982, which we found to be not worth keeping as it would limit more of our dataset for an unneeded value.


### Filtering Data: Cumulative Past \~25 Season Player Data

In addition to filtering to look only into post-1990 games and players that only played 40+ games in a specific season, we removed many of the categorical variables except for their positions, as we believed they would just hinder our machine learning algorithm's ability to interpret the data on its own. Additionally, we would look for outliers in our data (using the 1.5IQR from first and third quartile rule) to remove outliers. Sometimes, data would be distributed more widely, and so we would instead manually remove extreme (irregular) values. 


### Data Calculation: Top X Players From A Team In A Given Season

Since we intend to use player stat predictions to predict team success, we needed a way to compare different players across different teams. We hypothesized that since only a certain number of players will play in any given game, the most relevant players to analyze would be the top 8 players for each team. We determined that minutes played would be the best metric to determine the top 8 players for each team, since playing more minutes would naturally mean a greater impact on a team’s performance. 

We used the Player per Game CSV file. To find the top 8 players for each team, we first isolated the player data for a specific season. We chose the 2023 season, since it’s the most recently completed season, and top players change every year. Then we sorted the dataframe by minutes played. We extracted the list of teams and for each team, we extracted the top 8 players with the most minutes played from that team. 

We then generated a pairplot showing the relationships between the stats of the top player and win-loss percentage for each team, and then we looked at the pairplot for the 8th best player, to see how depth plays a role. Please see the notebook for these visualizations.


### Normalization, Standardization, and Data Transformation

Much of the dataset had normalized features - per 100 possessions, being a percentage, the NBA metric was within a certain range or scaled, etc. Along the way, we standardized / normalized / scaled data. Please see the notebook for these calculations. 


### Data Encoding

We one-hot-encoded player position to better supply data to any models for this project. For the most part, we did not encode most variables, since the value we are trying to predict, win-loss percentage, is already a number. Many of the input statistics / features are also numerical, so at most we’ll just have to normalize or standardize. We may need to one-hot encode some numerical data into categories / brackets, but that also likewise does not require data encoding.


## References

<https://www.basketball-reference.com/>
