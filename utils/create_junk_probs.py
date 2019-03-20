gameFile = open("2018_2019_allgames.csv", "r")
outFile = open("2018_2019_junk_probs.csv", "w")
k1 = 0
k2 = 0
for l in gameFile:
    line = l.strip()
    tokens = line.split(",")
    if tokens[0] == "Team 1":
        k1 = tokens.index("Kaggle ID 1")
        k2 = tokens.index("Kaggle ID 2")
    else:
        outFile.write("2019_" + tokens[k1] + "_" + tokens[k2] + ",0.5\n")

gameFile.close()
outFile.close()
