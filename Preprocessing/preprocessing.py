# !usr/bin/env python

import pickle
import numpy as np
import pandas as pd


LACTOCOCCUS_TAX = {
    "GCF_001622305.1 Lactococcus lactis subsp. cremoris strain=P7266, ASM162230v1": "L. cremoris",
    "GCF_002078765.2 Lactococcus lactis subsp. cremoris strain=UC109, ASM207876v2": "L. cremoris",
    "GCF_900099625.1 Lactococcus lactis strain=ATCC 19435, IMG-taxon 2597490350 annotated assembly": "L. lactis"
}

def filter_vmax(filepath, condition, tax_cluster, taxonomy=None):
    vmax_df = pd.read_csv(filepath, index_col=0)
    vmax_df["index"] = vmax_df["Plate"] + "_" + vmax_df["well"].map(str) +\
        vmax_df["chcc"].map(str)
    
    vmax_filtered = vmax_df[vmax_df[".map_name"].\
                        map(lambda x: x[:-1]) == condition]
    if tax_cluster:
        cluster_dict = get_cluster_dict(tax_cluster)
        vmax_filtered = vmax_filtered[vmax_filtered["chcc"].isin(cluster_dict)]
        vmax_filtered["cluster"] = vmax_filtered["chcc"].map(cluster_dict)

    vmax_filtered["salt"] = vmax_df[".map_name"].map(lambda x: x[-1] == "4")*1
    vmax_filtered = vmax_filtered[vmax_filtered["velocity"] < 0]

    if taxonomy:
        tax = get_taxonomy(taxonomy)
        vmax_filtered['taxonomy'] = vmax_filtered['chcc'].map(tax)

    return vmax_filtered

def get_cluster_dict(tax_cluster):
    cluster_df = pd.read_csv(tax_cluster, index_col=0)
    cluster_dict = dict(zip(
        cluster_df["chcc"].map(lambda x: int(x[4:])), 
        cluster_df["cluster"]))

    return cluster_dict

def get_taxonomy(tax_obj):
    tax_file = open(tax_obj,'rb')
    taxonomy = pickle.load(tax_file)
    tax_file.close()

    tax_dict = {id: LACTOCOCCUS_TAX[spec] for id, spec in taxonomy.items() if spec in LACTOCOCCUS_TAX}

    return tax_dict

def save_count_matrix(df, filename):
    np.savez_compressed(
        filename, counts=df.values, index=df.index, columns=df.columns)
    
def load_count_matrix(filename):
    data = np.load(filename, allow_pickle=True)

    return pd.DataFrame(
        data["counts"], index=data["index"], columns=data["columns"])