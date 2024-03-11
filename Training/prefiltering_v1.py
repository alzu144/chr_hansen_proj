# !usr/bin/env python

import argparse
import pandas as pd
from scipy import sparse

# user-defined scrips
PROJECT_DIR = "Updated project path"

import sys
sys.path.append(PROJECT_DIR)
from Preprocessing import load_count_matrix, save_count_matrix
from Filtering import MICC

DATA_PATH = PROJECT_DIR + "Data/Processed/"

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-k", "--kmer-data", type=str,
    help="file with the kmer data"
)
parser.add_argument(
    "-s", "--sparse-data", type=str,
    help="sparse matrix of the x train partition of the kmer data"
)
parser.add_argument(
    "-y", "--y-data", type=str,
    help="name of the csv file containing the training labels (y_train)"
)
parser.add_argument(
    "-o", "--output-data", type=str,
    help="filtered kmer data"
)
cli_args = parser.parse_args()


# load in data
print("Reading sparse matrix.")
X_train_sparse = sparse.load_npz(DATA_PATH + cli_args.sparse_data)
print("Finished reading sparse matrix.\n")

kmer_data = load_count_matrix(DATA_PATH + cli_args.kmer_data)
y_train = pd.read_csv(DATA_PATH + cli_args.y_data)

# filtering
print("Started MICC.")
micc_indexes = MICC(X_train_sparse, y_train, 0.03)
print("Finished MICC.\n")

kmer_data_filtered = kmer_data.iloc[:, micc_indexes]

# save filtered data
save_count_matrix(kmer_data_filtered, DATA_PATH + cli_args.output_data)
