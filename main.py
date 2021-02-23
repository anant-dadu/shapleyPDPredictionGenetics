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

features_starts = ['ENSG', 'rs', 'ENSG_rs', 'all_features']
feat1 = [i for i in k if i[:4]=='ENSG']
feat2 = [i for i in k if i[:2]=='rs']
feat3 = [i for i in k if i[:2]=='rs' or i[:4]=='ENSG']
feat4 = k
names = ['ENSG', 'rs', 'rs_ENSG', 'all']
import pickle
for e, feat in enumerate([feat1, feat2, feat3, feat4]):
    obj = ShapleyExplainations()
    train = obj.trainXGBModel(ppmi_traineddata_df, feat, 'PHENO')
    with open('trainXGB_gpu_{}.model'.format(names[e]), 'wb') as f:
        pickle.dump(train, f)

DF = {}
RES = {}
for name in names:
    with open('trainXGB_gpu_{}.model'.format(name), 'rb') as f:
        temp = pickle.load(f)
    DF[name] = list(temp[3]['ID_test'])
    RES[name] = (temp[3]['AUC_train'], temp[3]['AUC_test'])
    print (name, RES[name])

with open('trainXGB_gpu.aucs', 'wb') as f:
        pickle.dump(RES, f)

import pyensembl
ensembl = pyensembl.EnsemblRelease(100)

from tqdm import tqdm
dict_map_result = {}
for value in tqdm(temp[1]['X_train'].columns): # Feature name is a column of ENSGs.
    if value[:4] == 'ENSG':
        dict_map_result[value] = ensembl.gene_name_of_gene_id(value)

with open('ENSG_gene.mapping', 'wb') as f:
        pickle.dump(dict_map_result, f)