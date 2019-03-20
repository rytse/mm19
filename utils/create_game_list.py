#Generate all possible games given a list of teams and stats

inFile = open("2018_2019_data.csv", "r")
outFile = open("2018_2019_allgames.csv", "w")
varsToUse = ["Kaggle ID", "Region", "Seed", "Adj Offensive Efficiency", "Adj Defensive Efficiency", "Points Allowed Per Game", "Turnovers Per Game", "Wins Last 10 Games"]
tokensToUse = []
teamStatDict = {}
teams = []

for l in inFile:
    line = l.strip()
    tokens = line.split(",")
    if tokens[0] == "School Name":
        tokensToUse = [tokens.index(x) for x in varsToUse]
        continue
    else:
        team = tokens[0]
        stats = [tokens[x] for x in tokensToUse]
        teams.append(team)
        teamStatDict[team] = stats
inFile.close()
teams.sort()

outFile.write("Team 1,Team2,")
outFile.write(",".join([x + " Winner" for x in varsToUse]) + ",")
outFile.write(",".join([x + " Loser" for x in varsToUse]) + "\n")

for i in range(len(teams)):
    for j in range(i+1, len(teams)):
        outFile.write(teams[i] + "," + teams[j] + "," + ",".join(teamStatDict[teams[i]]) + "," + ",".join(teamStatDict[teams[j]]) + "\n")
outFile.close() 
