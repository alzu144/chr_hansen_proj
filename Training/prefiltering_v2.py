# !usr/bin/env python

import argparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# user-defined scrips
PROJECT_DIR = "XXX"

import sys
sys.path.append(PROJECT_DIR)
from Preprocessing import filter_vmax, load_count_matrix, save_count_matrix
from Filtering import MICC

DATA_PATH = PROJECT_DIR + "Data/Processed/"

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
    "-o", "--output-data", type=str,
    help="filtered kmer data"
)
cli_args = parser.parse_args()


# load in data
print("Preprocessing data.")
vmax_data = filter_vmax(
    filepath=DATA_PATH + cli_args.acid_data, 
    condition="salt tolerance_b-milk_30_", 
    tax_cluster=DATA_PATH + "jaccard_similarity_clusters_149.csv",
    taxonomy=DATA_PATH + "gtdb_taxonomy.obj")

kmer_data = load_count_matrix(DATA_PATH + cli_args.kmer_data)
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

X_train = kmer_data.loc[train_diff_df['chcc'].map(int)]

y_train = train_diff_df['velocity_x']
y_train.index = train_diff_df['chcc'].map(int)

print("Finished preprocessing data.\n")

# filtering
print("Started MICC.")
micc_indexes = MICC(X_train, y_train, 0.03)
print("Finished MICC.\n")

kmer_data_filtered = kmer_data.iloc[:, micc_indexes]

# save filtered data
save_count_matrix(kmer_data_filtered, DATA_PATH + cli_args.output_data)
