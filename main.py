from shapley_plots import ShapleyExplainations
import pandas as pd
import numpy as np
import os

ppmi_traineddata_path = "../data/PPMI_Only_Combined_Omics_v2.dataForML.h5"
ppmi_traineddata_path = "data_updated/PPMI-genetic_p1E5_omic_p1E2.dataForML.h5"
column_name_path = "../data/validation_PDBP.finalHarmonizedCols_toKeep.txt"
column_name_path = "data_updated/validate-PDBP-genetic_p1E5_omic_p1E2.finalHarmonizedCols_toKeep.txt"
with open(column_name_path, 'r') as the_file:
    st = the_file.read().strip().split('\n')
    d = {i:1 for i in st}
    k = [i for i in st if i not in ['ID', 'PHENO']]

ppmi_traineddata_df = pd.read_hdf(ppmi_traineddata_path, key='dataForML', mode='r')
matchingCols_file = open(column_name_path, "r")
matching_column_names_list = matchingCols_file.read().splitlines()
ppmi_traineddata_df = ppmi_traineddata_df[np.intersect1d(ppmi_traineddata_df.columns, matching_column_names_list)]
obj = ShapleyExplainations()
train = obj.trainXGBModel(ppmi_traineddata_df, k, 'PHENO')
import pickle
with open('trainXGB_new_data_gpu.model', 'wb') as f:
        pickle.dump(train, f)
