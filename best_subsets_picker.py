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

# Load data (training and validation)
train_raw_teamstats = pd.read_excel('cts.xlsx', sheet_name='2017-2018')
train_raw_games = pd.read_csv('clean_2017-2018_combo.csv')
val_raw_teamstats = pd.read_excel('cts.xlsx', sheet_name='2016-2017')
val_raw_games = pd.read_csv('clean_2016-2017_combo.csv')

# Clean all the bloody whitespace
train_raw_teamstats = train_raw_teamstats.rename(columns=lambda x: x.strip())
train_raw_games = train_raw_games.rename(columns=lambda x: x.strip())
val_raw_teamstats = val_raw_teamstats.rename(columns=lambda x: x.strip())
val_raw_games = val_raw_games.rename(columns=lambda x: x.strip())

# Get the list of all stats
VARS = list(train_raw_teamstats)
VARS.remove('Team Name')

def train_on_tk(v2u, rbest, obest, wbest):
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

	# Convert lists to np array
	X = np.array(Xs)
	y = np.array(ys)

	# Load each training game into a np array that will be the
	# validation data set
	Xvs = []
	yvs = []
	for idx, row in val_raw_games.iterrows():
		winner = row['Winner']
		loser = row['Loser']

		try:
			w_v = np.array(row[winner_vars])
			l_v = np.array(row[loser_vars])
			Xvs.append(np.array([w_v, l_v]).flatten())
			yvs.append(np.array([row['Winner Points'] - row['Loser points']]))
			Xvs.append(np.array([l_v, w_v]).flatten())
			yvs.append(np.array([row['Loser points'] - row['Winner Points']]))
		except:
			print(winner)
			print(loser)

	Xv = np.array(Xvs)
	yv = np.array(yvs)

	# Train a Gaussian Process Regression model
	kernel = obest * RBF(rbest) + WhiteKernel(wbest)
	gp = GaussianProcessRegressor(kernel,
								n_restarts_optimizer=10,
								normalize_y=True)
	gp.fit(X, y)
	rep = gp.score(Xv, yv)

	# Train the linear model on the same data
	lm = linear_model.LinearRegression(fit_intercept=False, normalize=True)
	lm.fit(X, y)
	lm.score(Xv, yv)

	gp_y_pred = []
	lm_y_pred = []
	gp_sd = []
	tys = []
	for idx, row in val_raw_games.iterrows():
		winner = row['Winner']
		loser = row['Loser']

		try:
			w_v = np.array(row[winner_vars])
			l_v = np.array(row[loser_vars])
			Xvs.append(np.array([w_v, l_v]).flatten())
			yvs.append(np.array([row['Winner Points'] - row['Loser points']]))

			gp_y_pred_, gp_sd_ = gp.predict(np.array([w_v, l_v]).flatten().reshape(1, -1), return_std=True)
			gp_y_pred.append(gp_y_pred_)
			gp_sd.append(gp_sd_)
			lm_y_pred.append(lm.predict(np.array([w_v, l_v]).flatten().reshape(1, -1)))
			tys.append(row['Winner Points'] - row['Loser points'])
		except:
			print(winner)
			print(loser)

	gp_y_pred=np.array(gp_y_pred).flatten()
	lm_y_pred=np.array(lm_y_pred).flatten()
	gp_sd = np.array(gp_sd).flatten()
	tys = np.array(tys).flatten()

	correct = 0
	for i in range(len(tys)):
		if tys[i] >= 0:
			if lm_y_pred[i] >= 0:
				correct += 1
		else:
			if lm_y_pred[i] < 0:
				correct += 1

	correct = 0
	for i in range(len(tys)):
		if tys[i] >= 0:
			if gp_y_pred[i] >= 0:
				correct += 1
		else:
			if gp_y_pred[i] < 0:
				correct += 1

	y_pred_probs = 1 - norm.cdf(-gp_y_pred / gp_sd)
	ll = log_loss(np.array(tys > 0, dtype=np.int), np.array([1 - y_pred_probs, y_pred_probs]).T, labels=[0, 1])

	return ll

RBF_LENGTH_SCALE = 54.5618245614035
RBF_KERNEL_SCALE = 18.538748538011692
NOISE_LENGTH_SCALE = 84.34343436464646

og_vars = ['Adj Off Efficiency', 'Adj Off-Def', 'Adj Def Efficiency', 'FG%', 'Strength of Schedule', 'Avg. Scoring margin', 'Rebounds', 'Wins Last 10 Games ','Turnovers per game']#$, 'Coach Record']

ll = train_on_tk(og_vars, RBF_LENGTH_SCALE, RBF_KERNEL_SCALE, NOISE_LENGTH_SCALE)
print('Log Loss: ' + str(ll))
