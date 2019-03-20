inFile = open("2019_noisy_out.csv", "r")
outFile = open("2019_noisy_out_fix.csv", "w") 
for l in inFile:
    line = l.strip()
    tokens = l.split(",")
    tokens2 = tokens[0].split("_")
    team1 = tokens2[1]
    team2 = tokens2[2]
    if int(team1) > int(team2):
        outFile.write(tokens2[0] + "_" + team2 + "_" + team1 + "," + str(1-float(tokens[1])) + "\n")
    else:
        outFile.write(line + "\n")
