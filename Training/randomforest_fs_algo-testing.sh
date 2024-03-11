for i in {1..10}
do
    python randomforest_featureselection.py \
        -a "Vmax.csv" \
        -k "11mer_matrix_sample.npz"
done