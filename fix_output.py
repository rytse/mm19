import sys

inFile = open("2019_clean_out.csv", "r")
outFile = open("2019_clean_out_fix.csv", "w")
outFile.write("ID,Pred\n")
count = 0
for l in inFile:
    line = l.strip()
    tokens = line.split(",")
    tokens2 = tokens[0].split("_")
    team1 = tokens2[1]
    team2 = tokens2[2]
    if int(team1) > int(team2):
        outFile.write(tokens2[0] + "_" + team2 + "_" + team1 + "," + str(1-float(tokens[1])) + "\n")
        count += 1
    else:
        outFile.write(line + "\n")
        count += 1
print(count, file=sys.stderr)
outFile.close()
