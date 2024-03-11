# !usr/bin/env python
# Created by "Alexander Zubov", 05/10/2023 ---------%
#       Email:DKTMPALZU@chr-hansen.com              %
# --------------------------------------------------%

import argparse
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit

# user-defined scrips
PROJECT_DIR = "/home/dktmpalzu/lactococcus_proj/"

import sys
sys.path.append(PROJECT_DIR)
from Preprocessing import filter_vmax, load_count_matrix

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
    "-s", "--sparse-data", type=str,
    help="name of the output npz file containing the sparse matrix"
)
parser.add_argument(
    "-y", "--y-data", type=str,
    help="name of the output csv file containing the training labels (y_train)"
)
cli_args = parser.parse_args()


# load in data
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

# create and save sparse matrix
X_train_sparse = sparse.csr_array(X_train.values)

sparse.save_npz(DATA_PATH + cli_args.sparse_data, X_train_sparse)

# save corresponding y_train data also
y_train.to_csv(DATA_PATH + cli_args.y_data, index=False)