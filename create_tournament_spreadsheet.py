import sys

#statName = sys.argv[1]
#gameName = sys.argv[2]
#outName = sys.argv[3]
statName = "2017-2018_data.csv"
gameName = "2017-2018_games.csv"
outName = "2017-2018_combo.csv"

statFile = open(statName, "r")
statDict = {}
for l in statFile:
    line = l.strip()
    tokens = line.split(",")
    team = tokens[0]
    statDict[team] = tokens[1:]
statFile.close()

outFile = open(outName, "w")

outFile.write("Winner,Loser,Winner Points,Loser points,OTs,")
outFile.write(",".join([x + " Winner" for x in statDict["Team Name"]]) + ",")
outFile.write(",".join([x + " Loser" for x in statDict["Team Name"]]) + "\n")

gameFile = open(gameName, "r")
for l in gameFile:
    line = l.strip()
    tokens = line.split(",")
    if tokens[0] == "Schl":
        continue
    #print(tokens, file=sys.stderr)
    team1 = tokens[0]
    team2 = tokens[1]
    pts1 = tokens[2]
    pts2 = tokens[3]
    ot = "0"
    if len(tokens) > 4 and len(tokens[4]) > 0:
        ot = tokens[4]
    outFile.write(",".join([team1, team2, pts1, pts2, ot]) + ",")
    if team1 in statDict.keys():
        outFile.write(",".join(statDict[team1]) + ",")
    else:
        print("Missing team:", team1, file=sys.stderr)
        outFile.write(","*len(statDict["Team Name"]))
    if team2 in statDict.keys():
        outFile.write(",".join(statDict[team2]) + "\n")
    else:
        print("Missing team:", team2, file=sys.stderr)
        outFile.write(","*(len(statDict["Team Name"])-1) + "\n")

gameFile.close()
outFile.close()
