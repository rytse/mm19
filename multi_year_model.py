import numpy as np
import pandas as pd
from scipy.stats import norm, zscore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics import log_loss
import itertools
import thutil

np.random.seed(1)

V2U = ['Coach Tourney Appearances',
		'FG%',
		'Adj Off Efficiency',
		'Adj Def Efficiency',
		'Strength of Schedule',
		'Win %',
		'Major Conference',
		'Turnovers per game',
		'Wins Last 10 Games',
		'Coach Record',
		'Seed',
		'Won a Major Conference',
		'Rebounds']

#V2U = ['Adj Off Efficiency', 'Adj Def Efficiency', 'SRS']

def flip_row(rep):
	flip = np.random.random() > 0.5

	if flip:
		copy = rep.copy()
		for name in rep.index:
			if name.endswith('Loser'):
				rep[name] = copy[name[0:-5] + 'Winner']
			elif name.endswith('Winner'):
				rep[name] = copy[name[0:-6] + 'Loser']

		rep['Loser'] = rep['Loser'] + '_' + year
		rep['Winner'] = rep['Winner'] + '_' + year
		rep['pdiff'] = -(rep['Winner Points'] - rep['Loser points'])

	else:
		rep['Winner'] = rep['Winner'] + '_' + year
		rep['Loser'] = rep['Loser'] + '_' + year
		rep['pdiff'] = rep['Winner Points'] - rep['Loser points']

# Load all the game data
game_data = []
for year in range(2014, 2018+1):
	year = f'{year - 1}-{year}'
	rep = pd.read_csv(f'{year}_combo.csv')
	rep.rename(columns=lambda x: ' '.join(x.split()), inplace=True)
	rep[rep.select_dtypes(include=[np.number]).columns].apply(zscore)
	rep['pdiff'] = rep['Winner Points'] - rep['Loser points']
	rep['Winner'] = rep['Winner'] + '_' + year
	rep['Loser'] = rep['Loser'] + '_' + year
	game_data.append(rep)
game_data = pd.concat(game_data)

raw_names = [foo.strip() for foo in list(pd.read_csv('2014-2015_data.csv').columns)]
raw_names.remove('Team Name')

for i, row in game_data.iterrows():
	if np.random.random() > 0.5:
		for name in raw_names:
			vname_w = name + ' Winner'
			vname_l = name + ' Loser'
			wv = game_data.at[i, vname_w]
			game_data.at[i, vname_w] = game_data.at[i, vname_l]
			game_data.at[i, vname_l] = wv
		game_data.at[i, 'pdiff'] *= -1

def go(v2u, sig1, sig2, ls1, ls2, ws, wl):
	# Set up winner, loser names for the selection of vars
	v2u_mod = []
	for v in v2u:
		v2u_mod.append(v + ' Winner')
		v2u_mod.append(v + ' Loser')

	# Set up Gaussian Process with its kernel (don't train yet)
	rbf1 = sig1 ** 2 * RBF(length_scale=ls1)
	rbf2 = sig2 ** 2 * RBF(length_scale=ls2)
	white = ws ** 2 * WhiteKernel(noise_level=wl**2)
	kernel = rbf1 + rbf2 + white

	gp = GaussianProcessRegressor(kernel,
		#n_restarts_optimizer=9,
		normalize_y=True)

	lls = []
	for val_year in range(2014, 2018+1):
		val_year = f'{val_year - 1}-{val_year}'
		train = game_data[game_data['Winner'].str.endswith(val_year) == False]
		val = game_data[game_data['Winner'].str.endswith(val_year) == True]

		# Select the proper subsets of the dataframe
		X_train = train[v2u_mod]
		y_train = train['pdiff']
		X_val = val[v2u_mod]
		y_val = val['pdiff']

		gp.fit(X_train, y_train)	# train
		gp_y_pred, gp_sd = gp.predict(X_val, return_std=True)	# validate


		# Calculate validation log loss
		y_pred_probs = 1 - norm.cdf(-gp_y_pred / gp_sd)
		ll = log_loss(np.array(y_val > 0, dtype=np.int),
						np.array([1 - y_pred_probs, y_pred_probs]).T,
						labels=[0, 1])
		lls.append(ll)

	print(str(np.mean(lls)) + ', ' + str(v2u))
	return np.mean(lls)

sig1 = 3.2
sig2 = 3.2
ls1 = 93
ls2 = 93
ws = 6
wl = 6

#best_ll = 1
#best_v2u = None
tp = thutil.ThreadPool(8)
for nvars in range(3, 5+1):
#for nvars in range(1, 3):
	for v2u in set(itertools.combinations(V2U, nvars)):
		tp.add_task(go, *(v2u, sig1, sig2, ls1, ls2, ws, wl))

#print('Best V2U: ' + str(best_v2u))
#print('Best LL: ' + str(best_ll))
