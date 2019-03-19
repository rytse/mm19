import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import log_loss

np.random.seed(1)

V2U = ['Adj Off Efficiency', 'Adj Def Efficiency', 'Turnovers per game', 'Wins Last 10 Games',
		'Points Allowed Per Game']

for val_year in range(2014, 2018 + 1):
	train_years = list(range(2014, 2018+1))
	train_years.remove(val_year)
	train_years = [f'{train_year - 1}-{train_year}' for train_year in train_years]
	val_year = f'{val_year - 1}-{val_year}'

	# Load data
	train_games = []
	for train_year in train_years:
		rep = pd.read_csv(f'{train_year}_combo.csv')
		rep.rename(columns=lambda x: ' '.join(x.split()), inplace=True)
		rep['Winner'] = rep['Winner'] + '_' + train_year
		rep['Loser'] = rep['Loser'] + '_' + train_year
		train_games.append(rep)
	train_games = pd.concat(train_games)

	val_games = pd.read_csv(f'{val_year}_combo.csv')
	val_games.rename(columns=lambda x: ' '.join(x.split()), inplace=True)

	# Construct list of features to use
	winner_vars = []
	loser_vars = []
	for v in V2U:
		winner_vars.append(v + ' Winner')
		loser_vars.append(v + ' Loser')

	# Load each training game into a np array that will be the
	# training data set
	Xs = []  # input vars
	ys = []  # output var
	for idx, row in train_games.iterrows():
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
		n_restarts_optimizer=100,
		normalize_y=True)
	gp.fit(X, y)

	# Validate GPR
	gp_y_pred = []
	gp_sd = []
	tys = []
	for idx, row in val_games.iterrows():
		w_v = np.array(row[winner_vars])
		l_v = np.array(row[loser_vars])

		gp_y_pred_, gp_sd_ = gp.predict(np.array([w_v, l_v]).flatten().reshape(1, -1),
										return_std=True)
		gp_y_pred.append(gp_y_pred_)
		gp_sd.append(gp_sd_)
		tys.append(row['Winner Points'] - row['Loser points'])

	gp_y_pred = np.array(gp_y_pred).flatten()
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
	ll = log_loss(np.array(tys > 0, dtype=np.int),
					np.array([1 - y_pred_probs, y_pred_probs]).T,
					labels=[0, 1])
	print(f'Log Loss: {ll}')
