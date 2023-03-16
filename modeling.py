import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, average_precision_score, balanced_accuracy_score
import statsmodels.formula.api as smf
from catboost import Pool, CatBoostClassifier
import shap
from tkr_methods.data_processing import impute_missing_data, separate_cat_num_features, coerce_categorical_features, split_train_val


def get_eval_metrics(y_true, y_pred): 
    
    metrics = {} 
    metrics['logloss'] = log_loss(y_true, y_pred).round(4)
    metrics['accuracy'] = accuracy_score(y_true, y_pred>.5).round(4)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred>.5).round(4)
    metrics['auroc'] = roc_auc_score(y_true, y_pred).round(4)
    metrics['auprc'] = average_precision_score(y_true, y_pred).round(4)
    
    return metrics 



def fit_eval_logit(cat_features, num_features, train, val, other_formula=None, label='KR_LABEL', 
                   display_summary=True):
    
    feature_cols = cat_features + num_features
    train, val = impute_missing_data(train, val, feature_cols)
    
    num_formula = ' + '.join(num_features)
    formula = f"{label} ~ {num_formula}" 
    if len(cat_features) > 0:
        cat_formula = ' + '.join([f'C({col})' for col in cat_features])
        formula = f"{formula} + {cat_formula}" 
    if other_formula is not None:
        formula = f"{formula} + {other_formula}" 

    logit = smf.logit(formula, data=train).fit()
    if display_summary:
        display(logit.summary())
    
    preds = val.assign(Prediction = logit.predict(val))
    metrics = get_eval_metrics(y_true=preds[label], y_pred=preds['Prediction'])
    print(metrics)
    
    return logit, preds, metrics 



def get_shap(model, df_exp_test, max_shap_display=50):

    explainer = shap.TreeExplainer(model.model)
    data_pool = Pool(data=df_exp_test[model.feature_cols], cat_features=model.cat_features)
    full_shap = explainer.shap_values(data_pool)
    shap.summary_plot(full_shap, features=df_exp_test[model.feature_cols], max_display=max_shap_display)
    
    return full_shap



def fit_eval_catboost(cat_features, num_features, train_df, val_df, test_df=None, label='KR_LABEL', verbose=10, 
                      learning_rate=None, iterations=None,
                      plot_training=False, plot_shap=False, max_shap_display=50, display_importances=False):
    
    # subset out features and create data pools
    feature_cols = cat_features + num_features
    train_pool = Pool(train_df[feature_cols], train_df[[label]], cat_features=cat_features)
    val_pool = Pool(val_df[feature_cols], val_df[[label]], cat_features=cat_features)

    # fit model
    model = CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss', learning_rate=learning_rate,
                               early_stopping_rounds=40, use_best_model=True, random_seed=42)
    model.fit(train_pool, eval_set=val_pool, verbose=verbose, plot=plot_training)

    # shap values 
    explainer = shap.TreeExplainer(model)
    full_shap = explainer.shap_values(val_pool)
    if plot_shap:
        shap.summary_plot(full_shap, features=val_df[feature_cols], max_display=max_shap_display)

    # compute two types of feature importances     
    shap_importances = pd.DataFrame(full_shap, columns=feature_cols).abs().mean().rename('ShapImportance')    
    lfc_importances = model.get_feature_importance(data=val_pool, type='LossFunctionChange', prettified=True)
    lfc_importances = lfc_importances.rename(columns={'Importances': 'LossFuncChgImportance'})
    importances = lfc_importances.merge(shap_importances, how='left', left_on='Feature Id', right_index=True)
    if display_importances:
        display(importances)
        
    # val set is used as test set during feature selection and tuning 
    if test_df is None: 
        test_df = val_df.copy() 
    test_pool = Pool(test_df[feature_cols], test_df[[label]], cat_features=cat_features)
    test_preds = test_df.assign(
        Probability = model.predict_proba(test_pool)[:,1],
        Prediction = model.predict(test_pool)
    )
    
    # metrics 
    metrics = get_eval_metrics(y_true=test_preds[label], y_pred=test_preds['Probability'])
    print(metrics)
    
    return test_preds, model, full_shap, importances, metrics 


def select_features(full_features, train, val, max_iter=100, eval_metric='auprc'): 
    
    results = {}
    current_features = full_features 
    
    for i in range(1, max_iter+1): 
        print(f"Iteration {i}") 
        current_num, current_cat = separate_cat_num_features(df=train, feature_cols=current_features)
        _, model, _, importances, metrics = fit_eval_catboost(
            train_df=train, val_df=val, plot_shap=False, verbose=False, 
            cat_features=current_cat, num_features=current_num)      
        loss = model.get_best_score()['validation']['Logloss'] 
        results[i] = {'features': current_features, 'eval_metric': metrics[eval_metric]}
        print(f"Using {len(current_features)} features, we obtain val loss of {loss}")
        
        keep_features = importances.query("LossFuncChgImportance > 0")['Feature Id'].unique().tolist()
        drop_features = [col for col in current_features if col not in keep_features]
        print(f"{len(drop_features)} features have negative importances and may be dropped in next iter: {drop_features}")
        print("")
        
        if len(drop_features) > 0: 
            current_features = keep_features 
        else: 
            break 
            
    results_df = pd.DataFrame.from_dict(results).T
    ascending = True if eval_metric == 'logloss' else False
    final_features = results_df.sort_values(by='eval_metric', ascending=ascending).iloc[0]['features']
    
    print(f"Finally {len(final_features)} are kept: {final_features}")
            
    return final_features 



def cv_catboost(cat_features, num_features, df, label='KR_LABEL', verbose=False, 
                learning_rate=None, iterations=None):
    
    preds = [] 
    metrics = []
    
    for fold in range(1, 5, 1):
        
        # predict OOS on one fold and perform 75/25 train-val split on the other folds 
        test_df = df.query("TrainFold==@fold")
        train_val_df = df.query("(Split=='train') & (TrainFold!=@fold)")        
        train_df, val_df = split_train_val(df=train_val_df, val_size=.25)
        
        # subset out features and create data pools
        feature_cols = cat_features + num_features
        train_pool = Pool(train_df[feature_cols], train_df[[label]], cat_features=cat_features)
        val_pool = Pool(val_df[feature_cols], val_df[[label]], cat_features=cat_features)    
        test_pool = Pool(test_df[feature_cols], test_df[[label]], cat_features=cat_features)    

        # fit model
        model = CatBoostClassifier(loss_function='Logloss', eval_metric='Logloss', learning_rate=learning_rate,
                                   early_stopping_rounds=40, use_best_model=True, random_seed=42)
        model.fit(train_pool, eval_set=val_pool, verbose=verbose, plot=False)
        
        # predict on test fold 
        fold_preds = test_df.assign(
            Probability = model.predict_proba(test_pool)[:,1],
            Prediction = model.predict(test_pool)
        )
        fold_preds['Fold'] = fold
        preds.append(fold_preds)
        
        # evaluate fold predictions  
        fold_metrics = get_eval_metrics(y_true=fold_preds[label], y_pred=fold_preds['Probability'])
        fold_metrics['Fold'] = fold
        metrics.append(fold_metrics)

    # concat preds and metrics 
    cv_preds = pd.concat(preds)
    cv_metrics = pd.DataFrame(metrics)
    cv_metrics.drop('Fold', axis=1).describe().T[['mean', 'std']]
    
    return cv_preds, cv_metrics 



def cv_logit(df, cat_features, num_features, other_formula=None):

    all_preds = []

    for fold in range(1, 5): 
        val = df.query("Fold==@fold")
        train = df.query("Fold!=@fold")
        logit, preds, metrics = fit_eval_logit(cat_features=cat_features, num_features=num_features, 
                                               other_formula=other_formula, train=train, val=val, 
                                               label='KR_LABEL', display_summary=False)
        all_preds.append(preds)
        
    out = pd.concat(all_preds)
        
    return out 