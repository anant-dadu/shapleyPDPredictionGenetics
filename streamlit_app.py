import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import shap
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def get_shapley_value_data(feature_set):
    with open('trainXGB_gpu_{}.model'.format(feature_set), 'rb') as f:
            train = pickle.load(f)
    dataset_type = '' 
    shap_values = np.concatenate([train[0]['shap_values_train'], train[0]['shap_values_test']], axis=0)
    X = pd.concat([train[1]['X_train'], train[1]['X_valid']], axis=0)
    exval = train[2]['explainer_train'] 
    auc_train = train[3]['AUC_train']
    auc_test = train[3]['AUC_test']
    ids = list(train[3]['ID_train'.format(dataset_type)]) + list(train[3]['ID_test'.format(dataset_type)])
    labels_pred = list(train[3]['y_pred_train'.format(dataset_type)]) + list(train[3]['y_pred_test'.format(dataset_type)]) 
    labels_actual = list(train[3]['y_train'.format(dataset_type)]) + list(train[3]['y_test'.format(dataset_type)]) 
    shap_values_updated = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
    train_samples = len(train[1]['X_train'])
    test_samples = len(train[1]['X_valid'])
    with open('ENSG_gene.mapping', 'rb') as f:
        dict_map_result = pickle.load(f)
    X.columns = ['({}) {}'.format(dict_map_result[col], col) if dict_map_result.get(col, None) is not None else col for col in list(X.columns)]
    import copy
    shap_values_updated = copy.deepcopy(shap_values_updated) 
    # patient_index = ['P{}'.format(e) for e, i in enumerate(ids)]
    import hashlib
    patient_index = [hashlib.md5(s.encode()).hexdigest() for e, s in enumerate(ids)]
    # zfile = ['{},{}'.format(ids[i], patient_index[i]) for i in range(len(ids))]
    # with open('mapping_id.txt', 'w') as f:
    #     f.write('\n'.join(zfile))
    # patient_index = copy.copy(ids)
    return (X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_updated, train_samples, test_samples)

max_width = 4000
padding_top = 10
padding_right = 10
padding_left = 10
padding_bottom = 10
COLOR = 'black'
BACKGROUND_COLOR = 'white'
st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
    )

import copy
st.title('SHAP values on PD prediction from Genetics')
with open('trainXGB_gpu.aucs', 'rb') as f:
    RES = pickle.load(f)

featset = ['ENSG', 'rs', 'rs_ENSG', 'all']
df_res = pd.DataFrame({'Feature Set': featset, 'Train AUC': ["{:.2f}".format(RES[i][0]) for i in featset], 'Test AUC':  ["{:.2f}".format(RES[i][1]) for i in featset]})
st.table(df_res.set_index('Feature Set'))
# feature_set_my = st.radio( "Select the feature set", ('ENSG', 'rs', 'rs_ENSG', 'all'), index=3, format_func=lambda x: [" Train AUC:{:.2f} Test AUC:{:.3f}".format(RES[i]) for i in x])
feature_set_my = st.radio( "Select the feature set", featset, index=3)


st.header("Introduction")
st.write(
    """
    SHAP is an unified approach to explain the output of any supervised machine learning model. SHAP values are generated based on the idea that the change of an outcome to be explained with respect to a baseline can be attributed in different proportions to the model input features. In addition to assigning an importance value to every feature based on SHAP values, it shows the direction-of-effect at the level of the model as a whole. Furthermore, SHAP values provide both the global interpretability (i.e. collective SHAP values can show how much each predictor contributes) and local interpretability that explain why a sample receives its prediction. We built a surrogate XGBoost classification model to understand each individual genetic features’ effect on the Parkinsons’ disease classification. We randomly split the dataset into training (70%) and test (30%) sets. The model is trained on the training set and the SHAP score on the validation data is analyzed to better understand the impact of features. Specifically, We used tree SHAP algorithm designed to provide human interpretable explanations for tree based learning models. We did not perform parameter tuning for the surrogate model. 
    """
)

st.header("Results")


data_load_state = st.text('Loading data...')
cloned_output = copy.deepcopy(get_shapley_value_data(feature_set_my))
data_load_state.text("Done Data Loading! (using st.cache)")
X, shap_values, exval, patient_index, auc_train, auc_test, labels_actual, labels_pred, shap_values_up, len_train, len_test = cloned_output 


col0, col00 = st.beta_columns(2)
with col0:
    st.subheader("Data Statistics")
    st.info ('Total Features: {}'.format(X.shape[1]))
    st.info ('Total Samples: {} (Train: {}, Test: {})'.format(X.shape[0], len_train, len_test))

with col00:
    st.subheader("XGBoost Model Performance")
    st.info ('AUC Train Score: {}'.format(round(auc_train,2)))
    st.info ('AUC Test Score:{}'.format( round(auc_test,2)))

st.write('***Class Labels:***', {'PD': 1.0, 'Control': 0.0})
st.subheader('Summary Plot')
st.write("""Shows top-20 features that have the most significant impact on the classification model. In the figure, it shows that University of Pennsylvania Smell Identification Test (UP SIT) is the most important factor. It also indicates that lower UPSIT feature (blue color) value corresponds to higher probability of PD, as most of the blue colored points lie on the right side of baseline. On the other end, for PRS90, lower expression values align with more healthy behaviour as blue colored points on the plot have negative impact on the model output. In this way, we can also observe that the directionality of different genetic features is not uniform.""")
shap_type = 'trainXGB'
col1, col2 = st.beta_columns(2)
with col1:
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns), show=False, max_display=50, order = shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs)# , return_objects=True 
    st.pyplot()
with col2:
    fig, ax = plt.subplots()
    shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0), show=False, max_display=50, order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs)
    st.pyplot(fig)

# shap.plots.beeswarm(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns), order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs, show=False, max_display=50)
    # r1 = shap.decision_plot(exval, np.copy(shap_values), list(X.columns), return_objects=True, show=False)
# shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0), show=False, max_display=50, order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0).abs)
# shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).mean(0), order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).abs.max(0), show=False, max_display=20)
# shap.plots.bar(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).abs.mean(0), order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).abs.max(0), show=False, max_display=20)
# fig, ax = plt.subplots()
# shap.plots.beeswarm(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns), order=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns).abs.max(0), show=False, max_display=20, link='logit')
# st.pyplot(fig)

st.subheader('Dependence Plots')
st.write("""We can observe the interaction effects of different features in for predictions. To help reveal these interactions dependence_plot automatically lists (top-3) potential features for coloring. 
Furthermore, we can observe the relationship betweem features and SHAP values for prediction using the dependence plots, which compares the actual feature value (x-axis) against the SHAP score (y-axis). 
It shows that the effect of feature values is not a simple relationship where increase in the feature value leads to consistent changes in model output but a complicated non-linear relationship.""")
feature_name = st.selectbox('Select a feature for dependence plot', options=list(X.columns))
inds = shap.utils.potential_interactions(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)[:, feature_name], shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns))

st.write('Top3 Potential Interactions for ***{}***'.format(feature_name))
col3, col4, col5 = st.beta_columns(3)
with col3:
    shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[0]]))
    st.pyplot()
with col4:
    shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[1]]))
    st.pyplot()
with col5:
    shap.dependence_plot(feature_name, np.copy(shap_values), X.copy(), interaction_index=list(X.columns).index(list(X.columns)[inds[2]]))
    st.pyplot()

st.subheader('Decision Plots')
st.write("""
    SHAP decision plots show how complex models arrive at their predictions (i.e., how models make decisions). 
    Each observation’s prediction is represented by a colored line.
    At the top of the plot, each line strikes the x-axis at its corresponding observation’s predicted value. 
    This value determines the color of the line on a spectrum. 
    Moving from the bottom of the plot to the top, SHAP values for each feature are added to the model’s base value. 
    This shows how each feature contributes to the overall prediction.
""")
# labels_pred_new = np.array(labels_pred, dtype=np.float)
labels_actual_new = np.array(labels_actual, dtype=np.float64)
y_pred = (shap_values.sum(1) + exval) > 0
misclassified = y_pred != labels_actual_new


st.write('#### Pathways for Prediction (Hierarchical Clustering)')
col3, col4, col5 = st.beta_columns(3)
with col3:
    st.write('Typical Prediction Path: Uncertainity (0.3-0.7)')
    r = shap.decision_plot(exval, np.copy(shap_values), list(X.columns), feature_order='hclust', return_objects=True, show=False)
    T = X.iloc[(np.array(labels_pred, dtype=np.float64) >= 0.3) & (np.array(labels_pred, dtype=np.float64) <= 0.7)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = np.copy(shap_values)[(np.array(labels_pred, dtype=np.float64) >= 0.3) & (np.array(labels_pred, dtype=np.float64) <= 0.7), :]
    fig, ax = plt.subplots()
    shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
    st.pyplot(fig)
    # fig, ax = plt.subplots()
    # r = shap.decision_plot(exval, np.copy(shap_values), list(X.columns), feature_order='hclust', return_objects=True)
    # st.pyplot()
with col4:
    st.write('Typical Prediction Path: Positive Class (>=0.95)')
    fig, ax = plt.subplots()
    T = X.iloc[np.array(labels_pred, dtype=np.float64) >= 0.95]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = np.copy(shap_values)[np.array(labels_pred, dtype=np.float64) >= 0.95, :]
    shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
    st.pyplot(fig)
with col5:
    st.write('Typical Prediction Path: Negative Class (<=0.05)')
    fig, ax = plt.subplots()
    T = X.iloc[np.array(labels_pred, dtype=np.float64) <= 0.05]
    import warnings
    with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           sh = np.copy(shap_values)[np.array(labels_pred, dtype=np.float64) <= 0.05, :]
    shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
    st.pyplot(fig)

st.write('#### Pathways for Prediction (Feature Importance)')
col31, col41, col51 = st.beta_columns(3)
with col31:
    st.write('Typical Prediction Path: Uncertainity (0.3-0.7)')
    r = shap.decision_plot(exval, np.copy(shap_values), list(X.columns), return_objects=True, show=False)
    T = X.iloc[(np.array(labels_pred, dtype=np.float64) >= 0.3) & (np.array(labels_pred, dtype=np.float64) <= 0.7)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = np.copy(shap_values)[(np.array(labels_pred, dtype=np.float64) >= 0.3) & (np.array(labels_pred, dtype=np.float64) <= 0.7), :]
    fig, ax = plt.subplots()
    shap.decision_plot(exval, sh, T, show=False, feature_order=r.feature_idx, link='logit', return_objects=True, new_base_value=0)
    st.pyplot(fig)
    # fig, ax = plt.subplots()
    # r = shap.decision_plot(exval, np.copy(shap_values), list(X.columns), feature_order='hclust', return_objects=True)
    # st.pyplot()
with col41:
    st.write('Typical Prediction Path: Positive Class (>=0.95)')
    fig, ax = plt.subplots()
    T = X.iloc[np.array(labels_pred, dtype=np.float64) >= 0.95]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh = np.copy(shap_values)[np.array(labels_pred, dtype=np.float64) >= 0.95, :]
    shap.decision_plot(exval, sh, T, show=False, link='logit',  feature_order=r.feature_idx, new_base_value=0)
    st.pyplot(fig)
with col51:
    st.write('Typical Prediction Path: Negative Class (<=0.05)')
    fig, ax = plt.subplots()
    T = X.iloc[np.array(labels_pred, dtype=np.float64) <= 0.05]
    import warnings
    with warnings.catch_warnings():
           warnings.simplefilter("ignore")
           sh = np.copy(shap_values)[np.array(labels_pred, dtype=np.float64) <= 0.05, :]
    shap.decision_plot(exval, sh, T, show=False, link='logit', feature_order=r.feature_idx, new_base_value=0)
    st.pyplot(fig)


st.write('#### Pathways for Misclassified Samples')
col6, col7 = st.beta_columns(2)
with col6:
    st.info('Misclassifications (test): {}/{}'.format(misclassified[len_train:].sum(), len_test))
    fig, ax = plt.subplots()
    r = shap.decision_plot(exval, shap_values[misclassified], list(X.columns), link='logit', return_objects=True, new_base_value=0)
    st.pyplot(fig)
with col7:
    # st.info('Single Example')
    sel_patients = [patient_index[e] for e, i in enumerate(misclassified) if i==1]
    select_pats = st.selectbox('Select misclassified patient id', options=list(sel_patients))
    id_sel_pats = sel_patients.index(select_pats)
    fig, ax = plt.subplots()
    shap.decision_plot(exval, shap_values[misclassified][id_sel_pats], X.iloc[misclassified,:].iloc[id_sel_pats], link='logit', feature_order=r.feature_idx, highlight=0, new_base_value=0)
    st.pyplot()



st.subheader('Data')
df = pd.DataFrame({'ID': patient_index,
                      'Actual Label': labels_actual,
                      'Predicted Label (PD probability)': labels_pred,
                      'Split': ['train']*len_train + ['test']*len_test,
                      'Correctness': [ not i for i in misclassified]
                     })
df['Actual Label'] = df['Actual Label'].map(lambda x: 'PD' if x==1 else 'HC')
df['Predicted Label (PD probability)'] = df['Predicted Label (PD probability)'].map(lambda x: round(float(x), 2))
# if st.checkbox('Show Misclassified Samples'):
#     df_up = df.copy()# [misclassified]
# else:
#     df_up = df.copy()
df_up = df.copy()
df_up = df_up.set_index('ID').sort_values(by=[ 'Split', 'Correctness'])
df_up = df_up[df_up['Split']=='test'].sort_values(by=['Split', 'Correctness', 'Predicted Label (PD probability)'])
selected = list(df_up[df_up['Correctness']==0].index)
st.write('#### {} Data Labels'.format('PPMI'.upper()))
st.write ('The table shows the predictions on the test data. Rows highlighted with light green colors are the misclassified examples.')
st.table(df_up.style.apply(lambda x: ['background: lightgreen'
                                  if (x.name in selected)
                                  else '' for i in x], axis=1))



st.subheader('Force Plots')
st.write("""
The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.
""")
patient_name = st.selectbox('Select patient id', options=list(patient_index))
# st.info('You selected ***{}***'.format(patient_name))
sample_id = patient_index.index(patient_name)
col8, col9 = st.beta_columns(2)
with col8:
    st.info('Actual Label: ***{}***'.format('PD' if labels_actual[sample_id]==1 else 'HC'))
    st.info('Predicted PD class Probability: ***{}***'.format(round(float(labels_pred[sample_id]), 2)))
with col9:
    shap.force_plot(exval, shap_values[sample_id,:], X.iloc[sample_id,:], show=False, matplotlib=True)
    st.pyplot()

col10, col11 = st.beta_columns(2)
with col10:
    fig, ax = plt.subplots()
    shap.decision_plot(exval, shap_values[sample_id], X.iloc[sample_id], link='logit', highlight=0, new_base_value=0)
    st.pyplot()



# labels_pred = np.array(labels_pred, dtype=np.float)



# shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)
# shap.dependence_plot("PRS90", shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns), X, interaction_index=list(X.columns).index('UPSIT'))

# inds = shap.utils.potential_interactions(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)[:, "PRS90"], shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns))
# st.write(inds)
# for i in range(3):
#     fig, ax = plt.subplots()
#     shap.plots.scatter(shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)[:,"PRS90"], color=shap.Explanation(values=np.copy(shap_values), base_values=np.array([exval]*len(X)), data=np.copy(X.values), feature_names=X.columns)[:,inds[i]], cmap=plt.get_cmap("cool"), show=False)
#     st.pyplot(fig)


# fig, ax = plt.subplots()
# shap.plots.beeswarm(shap_values_up, order=shap_values_up.abs.max(0), show=False, max_display=20)
# st.pyplot(fig)


# with col2:
# fig, ax = plt.subplots()
# shap.plots.beeswarm(shap_values_up.abs, order=shap_values_up.abs.max(0), color="shap_red", max_display=20)
# shap.plots.bar(shap_values_up.abs.mean(0), order=shap_values_up.abs.max(0), max_display=20)
# st.pyplot(fig)



# inds = shap.utils.potential_interactions(shap_values[:, "PRS90"], shap_values)
# 
# st.sidebar.write('It will display the shap values against the actual feature value for a specific feature. Check the box to see.')
# 
# if st.sidebar.checkbox('Dependence Plot'):
#     st.write('### Dependence Plot')
#     feature_name = st.sidebar.selectbox('Which feature do you want to display?', options=list(X.columns))
#     st.write('You selected ***{}***'.format(feature_name))
#     shap.dependence_plot(feature_name, shap_values, X, interaction_index=None)
#     st.pyplot()
# 
# st.sidebar.write('It will display the decision making procedure for the model for a chosen patient. Check the box to see.')
# if st.sidebar.checkbox('Force Plot'):
#     st.write('### Force Plot')
#     # sample_id = st.sidebar.slider('Choose sample index?', 0, len(patient_index)-1, 0)
#     # st.write('You selected ***{}***'.format(patient_index[sample_id]))
#     patient_name = st.sidebar.selectbox('Choose patient id?', options=list(patient_index))
#     st.write('You selected ***{}***'.format(patient_name))
#     sample_id = patient_index.index(patient_name)
#     st.write('Actual Label: ***{}***'.format('PD' if labels_actual[sample_id]==1 else 'HC'))
#     st.write('Predicted PD class Probability: ***{}***'.format(round(float(labels_pred[sample_id]), 2)))
#     shap.force_plot(exval, shap_values[sample_id,:], X.iloc[sample_id,:], show=False, matplotlib=True)
#     st.pyplot()
# 
# st.sidebar.write('It will display the patient id, actual class, and predicted probability of PD class. Check the box to see.')
# if st.sidebar.checkbox('Show Data Labels'):
#     df = pd.DataFrame({'ID': patient_index,
#                        'Actual Label': labels_actual,
#                        'Predicted Label (PD probability)': labels_pred,
#                       })
#     df['Actual Label'] = df['Actual Label'].map(lambda x: 'PD' if x==1 else 'HC')
#     df['Predicted Label (PD probability)'] = df['Predicted Label (PD probability)'].map(lambda x: round(float(x), 2))
#     st.write('### {} Data Labels'.format(dataset_type.upper()), df.set_index('ID').sort_values(by=['Actual Label', 'Predicted Label (PD probability)' ]))
# 
# 
# fig, ax = plt.subplots()
# shap.decision_plot(exval, shap_values, X.columns, show=False, link='logit')
# st.pyplot(fig)
# 
# 
# st.write(labels_pred[0])
# st.write(X.head())
# st.write(shap_values.shape)
# labels_pred = np.array(labels_pred, dtype=np.float)
# fig, ax = plt.subplots()
# T = X.iloc[labels_pred >= 0.98]
# import warnings
# with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        sh = shap_values[labels_pred >= 0.98, :]
# shap.decision_plot(exval, sh, T.columns, show=False, link='logit',  return_objects=True)
# st.pyplot(fig)
# 
# 
# # fig, ax = plt.subplots()
# # shap.dependence_plot("rs10835060", shap_values, X.columns, interaction_index="PRS90", show=False, cmap=plt.get_cmap("cool"))
# # st.pyplot(fig)
# 
# 
# inds = shap.utils.potential_interactions(shap_values[:, "PRS90"], shap_values)
# for i in range(5):
#     shap.plots.scatter(shap_values[:,"PRS90"], color=shap_values[:,inds[i]], cmap=plt.get_cmap("cool"))
