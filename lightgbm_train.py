import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from tqdm import tqdm, tqdm_notebook
import gc
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import random

def predict(X, df_model, mode='best_mean'):
    if mode == 'best_mean':
        y_preds = []
        for i in range(5):
            y_preds.append(df_model.loc[0, f'model_{i}'].predict(X))
        y_preds = np.mean(np.array(y_preds), axis=0)
    elif mode == 'ensemble_mean':
        y_preds = []
        for i in df_model.index:
            for j in range(5):
                y_preds.append(df_model.loc[i, f'model_{j}'].predict(X))
        y_preds = np.mean(np.array(y_preds), axis=0)
    elif mode == 'weighted_ensemble_mean':
        y_preds = []
#         model_weight = df_model['average_mcc'].apply(lambda a: a/df_model['average_mcc'].sum())
        model_weight = []
        for i in df_model.index:
            model_weight.append(1 + np.log10(df_model.shape[0] - i + 1))
        print(model_weight[:10])
        for i in df_model.index:
            for j in range(5):
                y_preds.append(
                    df_model.loc[i, f'model_{j}'].predict(X) *
                    model_weight[i]
                )
        y_preds = np.array(y_preds)
        y_preds = np.mean(y_preds, axis=0)
    else:
        raise ValueError("Mode isn't supported")
    
    return y_preds


def train_model(X, y, seed=42):
    param_dict = {'boosting_type': ['gbdt', 'dart'],
                    'learning_rate': [0.1, 0.03, 0.01],
                    'n_estimators': [100, 300],
                    'feature_fraction': [5/7 + 0.01, 1.0],
                    'lambda': [
                        # l1, l2
                        [0.0, 0.0],
                        [0.001, 0.01],
                        [0.01, 0.1],
                        [1.0, 0.01],
                    ]}
    param_key = list(param_dict.keys())
    param_item = list(param_dict.values())
    
    import itertools
    param_list = list(itertools.product(*param_item))
    df_model = pd.DataFrame(columns=[*param_key, *[f'model_{i}' for i in range(5)], *[f'model_{i}_auc' for i in range(5)], 'average_auc'])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for param in param_list[:5]:
        models = []
        ctr = 0
        auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(# fixed
                                        is_unbalance=True,
                                        seed=SEED,
                                        extra_trees=True,
                                        min_data_per_group=1,
                                        early_stopping_round=50,
                                        # tweak,
                                        **{
                                            param_key[0]:param[0],
                                            param_key[1]:param[1],
                                            param_key[2]:param[2],
                                            param_key[3]:param[3],
                                            'lambda_l1':param[4][0],
                                            'lambda_l2':param[4][0],
                                        }
                                )
            model.fit(X_train, y_train,
                    categorical_feature=cat_feature_idx,
                    eval_set=(X_val, y_val),
                    eval_metric='auc',
                    verbose=-1)

            y_val_pred = model.predict(X_val)
            auc_score = roc_auc_score(y_val, y_val_pred)

            models.append(model)
            auc_scores.append(auc_score)
            
        df_model.loc[ df_model.shape[0] ] = [*param,
                                            *models,
                                            *auc_scores,
                                            sum(auc_scores) / len(auc_scores)]
    return df_model

if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)


    ''' Preprocess Data'''
    train = pd.read_csv('./ISIC/train.csv')
    test = pd.read_csv('./ISIC/test.csv')
    sub = pd.read_csv('./ISIC/sample_submission.csv')
    for col in ['sex', 'anatom_site_general_challenge']:
        encoder = LabelEncoder()
        train[col].fillna('unknown', inplace = True)
        test[col].fillna('unknown', inplace = True)
        train[col] = encoder.fit_transform(train[col])
        test[col] = encoder.transform(test[col])
        
    age_approx = np.nanmean(np.concatenate([np.array(train['age_approx']), np.array(test['age_approx'])]))
    train['age_approx'].fillna(age_approx, inplace = True)
    test['age_approx'].fillna(age_approx, inplace = True)
    train['patient_id'].fillna('unknown', inplace = True)

    age_min_train = train.groupby('patient_id').age_approx.min()
    age_max_train = train.groupby('patient_id').age_approx.max()
    age_span_train = age_max_train - age_min_train
    train['age_min'] = train['patient_id'].apply(lambda pid: age_min_train[pid])
    train['age_max'] = train['patient_id'].apply(lambda pid: age_max_train[pid])
    train['age_span'] = train['patient_id'].apply(lambda pid: age_span_train[pid])


    age_min_test = test.groupby('patient_id').age_approx.min()
    age_max_test = test.groupby('patient_id').age_approx.max()
    age_span_test = age_max_test - age_min_test
    test['age_min'] = test['patient_id'].apply(lambda pid: age_min_test[pid])
    test['age_max'] = test['patient_id'].apply(lambda pid: age_max_test[pid])
    test['age_span'] = test['patient_id'].apply(lambda pid: age_span_test[pid])

    train = train.drop(['image_name', 'patient_id', 'diagnosis', 'benign_malignant'], axis=1)
    test = test.drop(['image_name', 'patient_id'], axis=1)
    print(train.head())
    print(test.head())

    # get index of categorical feature
    cat_feature = ['sex', 'anatom_site_general_challenge']
    cat_feature_idx = [train.columns.get_loc(ct) for ct in cat_feature]


    # split to X and y
    X = train.reset_index(drop=True)
    y = X['target']
    del X['target']

    X_test = test.copy()

    ''' Train Model '''
    df_model = None
    if not os.path.exists('./model.pkl'):
        df_model = train_model(X, y, seed=42)
        df_model = df_model.sort_values(by=['average_auc', 'boosting_type', 'learning_rate', 'n_estimators'], ascending=[False, True, True, True]).reset_index(drop=True)
        df_model.loc[:1000].to_pickle('model.pkl')
    else:
        df_model = pd.read_pickle('model.pkl')
        print(len(df_model))
    

    y_test_pred = predict(test, df_model, mode='best_mean')
    test['target'] = y_test_pred
    sub.target = test.target
    sub.to_csv('./submission.csv',index=False)
    print(sub.head())
    
    print(np.argmax(y_test_pred), y_test_pred[np.argmax(y_test_pred)])
    print(test.shape)
    a = test.iloc[0]
    print(a)
    # print(test.loc(np.argmax(y_test_pred)))
    lgb.plot_importance(df_model.loc[0, 'model_0'], ignore_zero=False, figsize=(16,9))
    plt.savefig('./importance_feature.png')

    "Test Single person"
    P_sex = 1
    P_age_approx = 70
    P_anatom_site_general_challenge = 5


    persons = {'sex':  [P_sex],
            'age_approx':  [P_age_approx],
            'anatom_site_general_challenge':  [P_anatom_site_general_challenge],
            'age_min':  [P_age_approx],
            'age_max':  [P_age_approx],
            'age_span':  [0]}
    select = pd.DataFrame(persons)
    print(select)
    y_test_pred = predict(select, df_model, mode='best_mean')
    print(y_test_pred)

    
    


    

    
