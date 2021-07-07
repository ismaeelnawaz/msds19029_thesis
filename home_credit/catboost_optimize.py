# -*- coding: utf-8 -*-

from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from skopt.space import Real, Categorical, Integer

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import pandas as pd

from time import time

from common import data_load_home_credit

def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))

    print('Best parameters:')
    print(best_params)
    return best_params

if __name__ == "__main__":
    train, test = data_load_home_credit('/media/ismaeel/Work/msds19029_thesis/dataset/home_with_missing.csv')
    feats = [f for f in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    cat_features = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
        'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_TYPE_SUITE', 'WALLSMATERIAL_MODE']

    roc_auc = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

    clf = CatBoostClassifier(thread_count=2,
                            loss_function='Logloss',
                            task_type = "GPU",
                            od_type = 'Iter',
                            verbose= False,
                            cat_features=cat_features
                            )
                        
    search_spaces = {'iterations': Integer(10, 1000),
                    'depth': Integer(1, 8),
                    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                    'random_strength': Real(1e-9, 10, 'log-uniform'),  
                    'bagging_temperature': Real(0.0, 1.0),
                    'border_count': Integer(1, 255),
                    'l2_leaf_reg': Integer(2, 30),
                    'scale_pos_weight':Real(0.01, 1.0, 'uniform')}

    opt = BayesSearchCV(clf,
                        search_spaces,
                        scoring=roc_auc,
                        cv=skf,
                        n_iter=100,
                        n_jobs=1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                        return_train_score=False,
                        refit=True,
                        optimizer_kwargs={'base_estimator': 'GP'},
                        random_state=42)

    best_params = report_perf(opt, train[feats], train['TARGET'],'CatBoost',
                          callbacks=[VerboseCallback(100), 
                                    DeadlineStopper(60*10)])