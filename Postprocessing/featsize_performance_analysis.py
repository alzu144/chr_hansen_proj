# !usr/bin/env python

import argparse
import glob
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

# user-defined scrips
PROJECT_DIR = "/home/dktmpalzu/lactococcus_proj/"

import sys
sys.path.append(PROJECT_DIR)
from Preprocessing import filter_vmax, load_count_matrix
from Models.tree_based import RF_DEFAULT_PARS

IN_DATA_PATH = PROJECT_DIR + "Data/Processed/"
OUT_DATA_PATH = PROJECT_DIR + "Data/Results/RandomForest_Training/FS_Testing_3/Featsize/"
RESULTS_DIR = "/home/dktmpalzu/lactococcus_proj/Data/Results/RandomForest_FeatureSelection/FS_Testing_3/"


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
    "--file-id", type=str, default="",
    help="an optional identifier for the output files"
)
cli_args = parser.parse_args()


################################################################################
##########                   Preprocess Input Data                    ##########
################################################################################

print("Preprocessing data.")
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
gss = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=0)

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

print("Finished preprocessing data.\n")


################################################################################
##########               Load in Global Best Position                 ##########
################################################################################

print("Loading in global best positions.")

n_epochs = 40
all_files = glob.glob(RESULTS_DIR + "*")
timestamp_pattern = r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{1})"
algorithm_pattern = r"FS_(.*?)_"

# list all files associated with the experiment
experiment_files = {}
for f in all_files:
    f = os.path.basename(f)

    timestamp_match = re.search(timestamp_pattern, f)
    timestamp_f = timestamp_match.group(1)
    algorithm_match = re.search(algorithm_pattern, f)

    if timestamp_match:
        if algorithm_match:
            algorithm = algorithm_match.group(1)
        else:
            continue
    
    if algorithm not in experiment_files:
        experiment_files[algorithm] = {"csv_files": [], "txt_files": [], "dates": []}
    
    if f.endswith(".txt"):
        experiment_files[algorithm]["txt_files"].append(f)
    if timestamp_f not in experiment_files[algorithm]["dates"]:
        experiment_files[algorithm]["dates"].append(timestamp_f)

algorithm_list = list(experiment_files.keys())

# load in the data - only the global best position is relevant here
experiment_data = {}
for algo in algorithm_list:
    # separate the .txt data
    txt_files = experiment_files[algo]["txt_files"]
    best_pos_files = []
    for f in txt_files:
        best_pos_match = re.search(r"global_best_pos", f)
        if best_pos_match:
            best_pos_files.append(f)

    # global best position
    algo_best_pos = np.zeros((n_epochs, kmer_data.shape[1]+1, len(best_pos_files)))
    for i in range(len(best_pos_files)):
        pos_i = np.loadtxt(RESULTS_DIR + best_pos_files[i])
        if len(pos_i) < n_epochs:
            pos_i = np.concatenate((np.zeros((n_epochs - len(pos_i), kmer_data.shape[1]+1)), pos_i), axis=0)
        algo_best_pos[:, :, i] = pos_i

    experiment_data[algo] = algo_best_pos

print("Finished loading in global best positions.\n")


################################################################################
##########                    Evaluate Positions                      ##########
################################################################################

print("Evaluating positions.")

for algo, best_pos in experiment_data.items():
    if algo == "AEO":
        algo_name = "AEO_DMOA"
    elif algo == "L":
        algo_name = "L_SHADE"
    print(f"\nAlgorithm: {algo_name}")
    
    for rep in range(0, best_pos.shape[2]):
        print(f"Replicate: {rep}")
        for epoch in range(0, best_pos.shape[0]):
            print(f"Epoch: {epoch}")
            best_pos_e = best_pos[epoch, :, rep].astype(int)

            X_train_iter = X_train.copy()
            X_test_iter = X_test.copy()
            X_eval_iter = X_eval.copy()

            X_train_iter = X_train_iter.iloc[:, best_pos_e == 1]
            X_test_iter = X_test_iter.iloc[:, best_pos_e == 1]
            X_eval_iter = X_eval_iter.iloc[:, best_pos_e == 1]

            # train model and get predictions
            rf_model = RandomForestRegressor(**RF_DEFAULT_PARS)

            print("Started training.")
            rf_model.fit(X_train_iter.values, y_train.values)
            print("Finished training.")

            y_test_hat = rf_model.predict(X_test_iter.values)
            y_eval_hat = rf_model.predict(X_eval_iter.values)

            # save results
            script_name = os.path.basename(__file__)
            out_file_id = "_".join([script_name, "subbestpos", algo_name, str(rep), str(epoch)])

            test_predict_df = pd.DataFrame({
                "chcc": y_test.index, "y_test": y_test.values, "y_test_hat": y_test_hat})
            test_predict_df.to_csv(OUT_DATA_PATH + out_file_id + "_test.csv")

            eval_predict_df = pd.DataFrame({
                "chcc": y_eval.index, "y_eval": y_eval.values,"y_eval_hat": y_eval_hat})
            eval_predict_df.to_csv(OUT_DATA_PATH + out_file_id + "_eval.csv")

print("\nFinished evaluating positions.\n")