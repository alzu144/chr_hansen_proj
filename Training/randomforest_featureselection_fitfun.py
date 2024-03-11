# !usr/bin/env python

import os
from datetime import datetime
import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

from mealpy.evolutionary_based.SHADE import L_SHADE
from mealpy.swarm_based.PSO import P_PSO

# user-defined scrips
PROJECT_DIR = "XXX"

import sys
sys.path.append(PROJECT_DIR)
from Preprocessing import filter_vmax, load_count_matrix
from Models.tree_based import RF_DEFAULT_PARS
from Metaheuristics import meta_fs_wrapper
from Metaheuristics.AEO_DMOA import AEO_DMOA
from Metaheuristics.SAGA import SAGA

IN_DATA_PATH = PROJECT_DIR + "Data/Processed/"
OUT_DATA_PATH = PROJECT_DIR + "Data/Results/RandomForest_FeatureSelection/"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-a", "--acid-data", type=str,
    help="file with the acidification data, such as vmax"
)
parser.add_argument(
    "-k", "--kmer-data", type=str,
    help="file with the kmer data"
)
parser.add_argument(
    "-e", "--experiment", type=str, default="",
    help="an optional experiment sub directory of the out data directory"
)
parser.add_argument(
    "-m", "--method", type=str, default="",
    help="the metaheuristic algorithm for optimization"
)
cli_args = parser.parse_args()

if cli_args.experiment:
    OUT_DATA_PATH += cli_args.experiment

# load in data
vmax_data = filter_vmax(
    filepath=IN_DATA_PATH + cli_args.acid_data, 
    condition="salt tolerance_b-milk_30_", 
    tax_cluster=IN_DATA_PATH + "jaccard_similarity_clusters_149.csv",
    taxonomy=IN_DATA_PATH + "gtdb_taxonomy.obj")

kmer_data = load_count_matrix(IN_DATA_PATH + cli_args.kmer_data)
if kmer_data.index.dtype == "O":
    kmer_data.index = kmer_data.index.map(lambda x: int(x[4:]))

# prepare data for machine learning
gss = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=23)

for train_test_idx, eval_idx in gss.split(
    X=kmer_data.loc[vmax_data['chcc'].map(int), :],
    y=vmax_data['velocity'], groups=vmax_data['cluster']):

    eval_data = vmax_data.iloc[eval_idx, :]

    for train_idx, test_idx in gss.split(
        X=kmer_data.loc[vmax_data.iloc[train_test_idx, :]['chcc'].map(int)],
        y=vmax_data.iloc[train_test_idx, :]['velocity'], 
        groups=vmax_data.iloc[train_test_idx, :]['cluster']):

        train_data = vmax_data.iloc[train_test_idx, :].iloc[train_idx, :]
        test_data = vmax_data.iloc[train_test_idx, :].iloc[test_idx, :]

print(
    'Evaluation Set:', len(eval_data), 
    '\nTesting Set:', len(test_data), 
    '\nTraining Set:', len(train_data)
)
print(
    len(set(eval_data['chcc'])) + len(set(test_data['chcc'])) + len(set(train_data['chcc'])), 
    'strains in total'
)
# check that test, train, and validation sets don't overlap (should print empty sets):
for i in ['chcc', 'cluster']:
    print(i, "test-eval intersection:", set(eval_data[i]).intersection(set(test_data[i])))
    print(i, "train-eval intersection:", set(eval_data[i]).intersection(set(train_data[i])))
    print(i, "train-test intersection:", set(test_data[i]).intersection(set(train_data[i])))

train_diff_df = pd.merge(
    train_data[train_data['salt']==1][['velocity','index', 'chcc']], 
    train_data[train_data['salt']==0][['velocity','index', 'chcc']], 
    on=['chcc','index']
)
test_diff_df = pd.merge(
    test_data[test_data['salt']==1][['velocity','index', 'chcc']], 
    test_data[test_data['salt']==0][['velocity','index', 'chcc']], 
    on=['chcc','index']
)
eval_diff_df = pd.merge(
    eval_data[eval_data['salt']==1][['velocity','index', 'chcc']], 
    eval_data[eval_data['salt']==0][['velocity','index', 'chcc']], 
    on=['chcc','index']
)

X_train = kmer_data.loc[train_diff_df['chcc'].map(int)]
X_train['normal_growth'] = train_diff_df['velocity_y'].values

y_train = train_diff_df['velocity_x']
y_train.index = train_diff_df['chcc'].map(int)

X_test = kmer_data.loc[test_diff_df['chcc'].map(int)]
X_test['normal_growth'] = test_diff_df['velocity_y'].values

y_test = test_diff_df['velocity_x']
y_test.index = test_diff_df['chcc'].map(int)

X_eval = kmer_data.loc[eval_diff_df['chcc'].map(int)]
X_eval['normal_growth'] = eval_diff_df['velocity_y'].values

y_eval = eval_diff_df['velocity_x']
y_eval.index = eval_diff_df['chcc'].map(int)


# define selector and start optimization
rf_model = RandomForestRegressor(**RF_DEFAULT_PARS)
if cli_args.method == "AEO_DMOA":
    hho_optimizer = AEO_DMOA(epoch=40, pop_size=10)
if cli_args.method == "L_SHADE":
    hho_optimizer = L_SHADE(epoch=40, pop_size=10)
if cli_args.method == "SAGA":
    hho_optimizer = SAGA(epoch=30, pop_size=10, init_pop_size=40, sa_epoch=5)
if cli_args.method == "PPSO":
    hho_optimizer = P_PSO(epoch=40, pop_size=10)

print("\nStarted feature selection.\n")
fitted_selector, _, _ = meta_fs_wrapper(
    X_train.values, y_train.values, X_test.values, y_test.values, rf_model, 
    hho_optimizer, out_path=OUT_DATA_PATH, fitting={"fit_weights": (0.99, 0.01)},
    mha="fitfun"
)
print("\nFinished feature selection.\n")

# get predictions with best feature subset
X_train_selected = fitted_selector.transform(X_train.values)
print("Started training with best features.")
rf_model.fit(X_train_selected, y_train)
print("Finished training.")

X_eval_selected = fitted_selector.transform(X_eval.values)
y_eval_hat = rf_model.predict(X_eval_selected)


# save results
timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
script_name = os.path.basename(__file__)
out_file_id = "_".join([timestamp, script_name])

eval_predict_df = pd.DataFrame({
    "chcc": y_eval.index, "y_eval": y_eval.values,"y_eval_hat": y_eval_hat})
eval_predict_df.to_csv(OUT_DATA_PATH + out_file_id + "_eval.csv")
