import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def split_train_val(df, label="KR_LABEL", val_size=.25, patient_id='ID'): 
    """ Splits train data into train/val for model training """ 
        
    # use only train data 
    out = df.query("Split=='train'")
    
    # create stratified group train/test split 
    out = out.set_index('Knee_ID')
    train_idx, test_idx, iter_df = append_stratified_group_split(
        df=out, label=label, test_size=val_size, group_col=patient_id, eligible_col='Eligible')
    out = out.assign(TrainSplit = lambda x: np.where(x.index.isin(train_idx), 'train', 
                                                     np.where(x.index.isin(test_idx), 'val', None)))
    
    # return train and val dfs 
    train_df = out.query("TrainSplit=='train'")
    val_df = out.query("TrainSplit=='val'")
    
    return train_df, val_df 


def append_stratified_group_split(df, label, test_size, group_col, eligible_col=None, 
                                  n_splits=1000, random_state=42):
    """ Borrows same function from DataPrep notebook used to split train/test. 
        Here we use it to split train into train/val. """

    # filter out non-eligible knees as defined in previous step 
    if eligible_col is not None: 
        df = df[df[eligible_col]==1]

    # initialize splitter 
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    groups = df[group_col]

    # for each split, compute the absolute difference in proportion of positive labels 
    results = [] 
    for i, (train_idx, test_idx) in enumerate(gss.split(X=df, groups=groups)):
        result = {} 
        result['train_idx'] = train_idx
        result['test_idx'] = test_idx
        result['train_pos'] = df.iloc[train_idx][label].mean() #df[df.index.isin(train_idx)][label].mean()
        result['test_pos'] = df.iloc[test_idx][label].mean() #df[df.index.isin(test_idx)][label].mean() 
        result['abs_diff_pos'] = np.abs(result['train_pos'] - result['test_pos'])
        results.append(result)

    # pick the split that gives the smallest difference in % of positive labels 
    out = pd.DataFrame(results)
    out = out.sort_values(by='abs_diff_pos', ascending=True)
    best = out.iloc[0]

    # return indices as knee ids 
    train_idx = df.iloc[best['train_idx']].index
    test_idx = df.iloc[best['test_idx']].index

    return train_idx, test_idx, out 



def impute_missing_data(train, val, feature_cols): 
    
    train = train.copy()
    val = val.copy()
    
    for col in feature_cols: 
        
        # for continuous features, take median 
        if train[col].nunique() > 8: 
            impute_val = train[col].median()
        # for other features, take mode 
        else: 
            impute_val = train[col].mode()[0]
        train[col] = train[col].fillna(impute_val)
        val[col] = val[col].fillna(impute_val)
        
    return train, val



def separate_cat_num_features(df, feature_cols):
    """ Given a dataframe, separate its columns into numeric vs. categorical features using dtype """
    
    # TODO: account for known categorical features 
    known_categorical = ['P02RACE', 'V00HLTHCAR', 'P02KRS3CV', 'V00MARITST','P02ELGRISK']
    
    # split num vs. cat 
    num_features = df[feature_cols].select_dtypes(include=np.number).columns.tolist() 
    num_features = [col for col in num_features if col not in known_categorical]
    cat_features = [col for col in df[feature_cols].columns if col not in num_features]
    
    return num_features, cat_features 


def coerce_categorical_features(df):
    """ Coerce all non-numeric columns as string to be treated as categorical features  """
    
    df = df.copy()
    _, cat_features = separate_cat_num_features(df=df, feature_cols=df.columns.tolist())
    for col in cat_features:
        df[col] = df[col].apply(str) 
        
    return df 
