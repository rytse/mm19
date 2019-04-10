import sys

probFile = open("2019_clean_out_fix.csv", "r")
idFile = open("2019_game_strings.txt", "r")
outFile = open("2019_clean_predictions.txt", "w")
probDict = {}
for line in probFile:
    tokens = line.strip().split(",")
    if len(tokens) == 2 and not(tokens[0] == "ID"):
        tokens2 = tokens[0].split("_")
        probDict[tokens[0]] = tokens[1]
        probDict[tokens2[0]+"_"+tokens2[2]+"_"+tokens2[1]] = str(1-float(tokens[1]))
probFile.close()

for line in idFile:
    string = line.strip()
    if string in probDict.keys():
        outFile.write(probDict[string] + "\n")
    else:
        print("Can't find:", string, file=sys.stderr)
idFile.close()
outFile.close()


        
