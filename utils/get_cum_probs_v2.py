import os, sys

slash = "/"
if os.name == "nt":
    slash = "\\"

teamFileName = ".." + slash + "2018_2019_data.csv" #2018-2019 data sheet from Stein
probFileName = ".." + slash + "2019_avg_out.csv" #Kaggle submission
outFileName = ".." + slash + "2019_avg_cps_2.csv" #Output
inFileName = ".." + slash + "2019_still_in_it.csv"
currentRound = 1 #1-based

# KaggleID (Str) --> Team Name
kaggleTeamDict = {} #TODO

#East, Midwest, South, West
#Region --> [[kaggleIDs(s)]]
regionSeedDict = {} #

nameIndex = 0
regionIndex = 1
kaggleIndex = 2
seedIndex = 3
teamFile = open(teamFileName, "r")
for l in teamFile:
    line = l.strip()
    tokens = line.split(",")
    if tokens[0] != "School Name":
        kaggleTeamDict[tokens[2]] = tokens[0]
        if not(tokens[1] in regionSeedDict.keys()):
            regionSeedDict[tokens[1]] = [[] for x in range(16)]
        regionSeedDict[tokens[1]][int(tokens[3])-1].append(tokens[2])
teamFile.close()            

#(team1, team2) --> Win Prob
winDict = {} 

probFile = open(probFileName, "r")
for l in probFile:
    line = l.strip()
    tokens = line.split(",")
    tokens2 = tokens[0].split("_")
    winDict[(tokens2[1], tokens2[2])] = float(tokens[1])
    winDict[(tokens2[2], tokens2[1])] = 1 - float(tokens[1])
    
    


#KaggleID --> prob.
roundProbs = [{}, {}, {}, {}, {}, {}, {}] #Note P(round 6) := P(winning it all)
#Also, rounds are numbered from zero
#Round 1(0) probabilites (temp.)
##for reg in regionSeedDict.keys():
##    for i in range(len(regionSeedDict[reg])):
##        if len(regionSeedDict[reg][i]) == 1:
##            roundProbs[0][regionSeedDict[reg][i][0]] = 1
##        else: #Assuming 2 teams
##            roundProbs[0][regionSeedDict[reg][i][0]] = winDict[(regionSeedDict[reg][i][0], regionSeedDict[reg][i][1])]
##            roundProbs[0][regionSeedDict[reg][i][1]] = winDict[(regionSeedDict[reg][i][1], regionSeedDict[reg][i][0])]

#Ignore teams that have lost
inCurrentRound = {}
inFile = open(inFileName, "r")
for l in inFile:
    line = l.strip()
    tokens = line.split(",")
    if tokens[0] != "School Name":
        for i in range(currentRound):
            roundProbs[i][tokens[1]] = float(tokens[i+2])
        inCurrentRound[tokens[1]] = float(tokens[currentRound+2])
inFile.close()
    
#Round 2-5 (1-4) probabilities:
for i in range(currentRound-1, 4): #i is current round, getting next round
    for reg in regionSeedDict.keys():
        #Calculate probabilities
        for j in range(2**(4-i)):
            k = 2**(4-i)-j-1
            #ASSUMPTION: Sum of Probs. of making it to this round for each seed is 1
            #This is the key part
            for t1 in regionSeedDict[reg][j]:
                cumProb = 0
                sumOppProbs = 0 #This handles if possible opponents are knocked out
                for t2 in regionSeedDict[reg][k]:
                    if inCurrentRound[t2]:
                        cumProb += roundProbs[i][t2] * winDict[(t1, t2)]
                        sumOppProbs += roundProbs[i][t2]
                print(sumOppProbs, file=sys.stderr)
                if sumOppProbs > 0:
                    cumProb /= sumOppProbs
                else:
                    cumProb = 1
                cumProb *= min(roundProbs[i][t1], inCurrentRound[t1])
                roundProbs[i+1][t1] = cumProb
        #Re-seed
        for j in range(2**(3-i)):
            regionSeedDict[reg][j].extend(regionSeedDict[reg].pop())

#Predicting last 2 (round 6/5):
#East vs West and South vs Midwest
                #Gross hardcoding
regList = ["East", "West", "Midwest", "South"]
regOppList = ["West", "East", "South", "Midwest"]
ewTeams = []
msTeams = []
for i in range(4):
    for t1 in regionSeedDict[regList[i]][0]: #Should have only one list
        cumProb = 0
        for t2 in regionSeedDict[regOppList[i]][0]:
            cumProb += roundProbs[4][t2] * winDict[(t1, t2)]
        cumProb *= roundProbs[4][t1]
        roundProbs[5][t1] = cumProb
        if i < 2:
            ewTeams.append(t1)
        else:
            msTeams.append(t1)

#Predicting winner (round 7/6):
for t1 in ewTeams:
    cumProb = 0
    for t2 in msTeams:
        cumProb += roundProbs[5][t2] * winDict[(t1, t2)]
    cumProb *= roundProbs[5][t1]
    roundProbs[6][t1] = cumProb
for t1 in msTeams:
    cumProb = 0
    for t2 in ewTeams:
        cumProb += roundProbs[5][t2] * winDict[(t1, t2)]
    cumProb *= roundProbs[5][t1]
    roundProbs[6][t1] = cumProb

#Output!!!
outFile = open(outFileName, "w")
outFile.write("Team,Round 1,Round 2,Round 3,Round 4,Round 5,Round 6,Round 7,\n")
for kid in kaggleTeamDict.keys():
    outFile.write(kaggleTeamDict[kid] + ",")
    for i in range(7):
        outFile.write(str(roundProbs[i][kid]) + ",")
    outFile.write("\n")
outFile.close()    
              
        
        
                
                

    

               
            

