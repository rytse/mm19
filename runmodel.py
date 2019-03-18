import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel
from sklearn import linear_model
from sklearn.metrics import log_loss

from scipy.stats import zscore, norm

import itertools

np.random.seed(1)

# Load data
train_raw_teamstats = pd.read_excel('cts.xlsx', sheet_name='2017-2018')
train_raw_games = pd.read_csv('clean_2017-2018_combo.csv')
val_raw_teamstats = pd.read_excel('cts.xlsx', sheet_name='2016-2017')
val_raw_games = pd.read_csv('clean_2016-2017_combo.csv')

# Clean data
train_raw_teamstats = train_raw_teamstats.rename(columns=lambda x: x.strip())
train_raw_games = train_raw_games.rename(columns=lambda x: x.strip())
val_raw_teamstats = val_raw_teamstats.rename(columns=lambda x: x.strip())
val_raw_games = val_raw_games.rename(columns=lambda x: x.strip())

# Construct list of features to use
v2u = ['Adj Off Efficiency', 'Adj Def Efficiency', 'Turnovers per game', 'Wins Last 10 Games',
           'Points Allowed Per Game']
winner_vars = []
loser_vars = []
for v in v2u:
	if v != 'Team Name':
		winner_vars.append(v + ' Winner')
		loser_vars.append(v + ' Loser')

# Load each training game into a np array that will be the
# training data set
Xs = []    # input vars
ys = []    # output var
for idx, row in train_raw_games.iterrows():
	w_v = np.array(row[winner_vars])
	l_v = np.array(row[loser_vars])
	Xs.append(np.array([w_v, l_v]).flatten())
	ys.append(np.array([row['Winner Points'] - row['Loser points']]))
	Xs.append(np.array([l_v, w_v]).flatten())
	ys.append(np.array([row['Loser points'] - row['Winner Points']]))
X = np.array(Xs)
y = np.array(ys)

# Train a Gaussian Process Regression (GPR) model
obest = 11.789993587770596
rbest = 93.72828049656303
wbest = 25.771604944753086
kernel = obest * RBF(rbest) + WhiteKernel(wbest)
gp = GaussianProcessRegressor(kernel,
			n_restarts_optimizer=10,
			normalize_y=True)
gp.fit(X, y)

# Validate GPR
gp_y_pred = []
gp_sd = []
tys = []
for idx, row in val_raw_games.iterrows():
	w_v = np.array(row[winner_vars])
	l_v = np.array(row[loser_vars])

	gp_y_pred_, gp_sd_ = gp.predict(np.array([w_v, l_v]).flatten().reshape(1, -1), return_std=True)
	gp_y_pred.append(gp_y_pred_)
	gp_sd.append(gp_sd_)
	tys.append(row['Winner Points'] - row['Loser points'])

gp_y_pred=np.array(gp_y_pred).flatten()
gp_sd = np.array(gp_sd).flatten()
tys = np.array(tys).flatten()

# Calculate percent of games predicted correctly
correct = 0
for i in range(len(tys)):
	if tys[i] >= 0:
		if gp_y_pred[i] >= 0:
			correct += 1
	else:
		if gp_y_pred[i] < 0:
			correct += 1

print(f'{correct / len(tys) * 100}% of games predicted correctly')

# Calculate validation log loss
y_pred_probs = 1 - norm.cdf(-gp_y_pred / gp_sd)
ll = log_loss(np.array(tys > 0, dtype=np.int), np.array([1 - y_pred_probs, y_pred_probs]).T, labels=[0, 1])
print(f'Log Loss: {ll}')
