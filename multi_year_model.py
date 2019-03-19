import numpy as np
import pandas as pd
from scipy.stats import norm, zscore
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics import log_loss

np.random.seed(1)

tmp='''
V2U = ['Adj Off Efficiency',
		'Adj Off-Def',
		'Adj Def Efficiency',
		'FG%',
		'Strength of Schedule',
		'Avg. Scoring margin',
		'Rebounds',
		'Wins Last 10 Games',
		'Turnovers per game',
		'Coach Record']#'''

#V2U = ['Adj Off Efficiency', 'Adj Def Efficiency', 'Turnovers per game', 'Wins Last 10 Games',
#		'Points Allowed Per Game']

V2U = ['Adj Off Efficiency', 'Adj Def Efficiency', 'SRS']

obest = 11.789993587770596
rbest = 93.72828049656303
wbest = 25.771604944753086

sig1 = 3.2
sig2 = 3.2
ls1 = 93
ls2 = 93
sigw = 6

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

v2u_mod = []
for v in V2U:
	v2u_mod.append(v + ' Winner')
	v2u_mod.append(v + ' Loser')

def go():
	# Set up Gaussian Process with its kernel (don't train yet)
	rbf1 = sig1 ** 2 * RBF(length_scale=ls1)
	rbf2 = sig2 ** 2 * RBF(length_scale=ls2)
	white = sigw ** 2 * WhiteKernel(noise_level=sigw**2)
	kernel = rbf1 + rbf2 + white

	gp = GaussianProcessRegressor(kernel,
		normalize_y=True)

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
		print(f'Log Loss: {ll}\n\n')


go()
