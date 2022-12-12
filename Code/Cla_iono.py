from sklearn.model_selection import train_test_split
import csv
import os
import gpboost as gpb
import numpy as np
import datetime
import pandas as pd
import CRPS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, log_loss, auc, roc_curve

path = 'C:/Users/localadmin/Desktop/Thesis/data'

path_ionosphere = path + '/ionosphere.csv'

df_iono = pd.read_csv(path_ionosphere, header=None)

col_names = ['class_target', 'a01',
             'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14',
             'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
             'a28', 'a29', 'a30', 'a31', 'a32', 'a33', 'a34']

list_key = []

for i in range(35):
    list_key.append(i)

dict_rename = dict(zip(list_key, col_names))

df_iono = df_iono.rename(columns=dict_rename)

df_iono = df_iono.dropna()

for i in range(4):
    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/Iono"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "\{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "\{}{}{}".format(date.hour,date.minute,date.second)

    os.mkdir(path)

    #Set likelihood (if regression = "gaussian", if categorical = "bernouli_probit" or "bernouli_logit"
    likelihood = 'bernoulli_probit'

    #Create first train params. Later they will be overwritten
    params = {'objective': likelihood,
              'learning_rate': 0.01,
              'max_depth': 3,
              'verbose': 0}

    if likelihood == "gaussian":
        num_boost_round = 50
        params['objective'] = 'regression_l2'
    if likelihood in ("bernoulli_probit", "bernoulli_logit"):
        num_boost_round = 500
        params['objective'] = 'binary'

    print('First params: ', params)



    #Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [0.01], 'min_data_in_leaf': [1,10,100],  #0.1,0.01  # 10,100
                    'max_depth': [1,5,10,-1]} #5,10,-1

    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_iono[['a01',
             'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14',
             'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
             'a28', 'a29', 'a30', 'a31', 'a32', 'a33']],
                                                        df_iono['class_target'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    col_X_train = ['a01',
             'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14',
             'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
             'a28', 'a29', 'a30', 'a31', 'a32', 'a33']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    #Create Model with GPBoost
    gp_model = gpb.GPModel(likelihood=likelihood,gp_coords=data_train.data)
    gp_model.set_optim_params({'optimizer_cov': 'nelder_mead'})

    #Hyperparametertuning
    opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                                 params=params,
                                                 num_try_random=None,
                                                 nfold=4,
                                                 gp_model=gp_model,
                                                 use_gp_model_for_validation=True,
                                                 train_set= data_train,
                                                 verbose_eval=1,
                                                 num_boost_round=1000,
                                                 early_stopping_rounds=10,
                                                 seed=1000,
                                                 train_gp_model_cov_pars=False)

    #Print statement to see best parameters
    print(opt_params)

    #Set parameters to results from above
    params = {'learning_rate': opt_params['best_params']['learning_rate'],
              'min_data_in_leaf': opt_params['best_params']['min_data_in_leaf'],
              'max_depth': opt_params['best_params']['max_depth']}

    print("Params set: ", params)

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    #Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'],
                    gp_model=gp_model,
                    train_gp_model_cov_pars=False)

    #Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    predictions = []

    for i in pred_resp['response_mean']:
        if i >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate ERR
    cf = confusion_matrix(y_true=y_test,y_pred=predictions)
    tn, fp, fn, tp = cf.ravel()
    ERR = (fp+fn) / (tn + fp + fn + tp)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, pred_resp['response_mean'])
    AUC_score = auc(fpr, tpr)

    #Calculate Logloss

    Logloss_score = log_loss(y_test,predictions)

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id','Date', 'ERR','Logloss','AUC', 'opt_params', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                            date.minute,
                                                            date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           ERR,
                                           Logloss_score,
                                           AUC_score,
                                           opt_params,
                                           'Gaussian Process Boosting']

    df_output['random_state'] = random_state

    path1 = path

    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/Iono/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    bst.save_model(path1 + "/model.json")

#
for i in range(4):
    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/Iono"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "/{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "/{}{}{}{}".format(date.hour,date.minute,date.second,date.microsecond)

    os.mkdir(path)

    #Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'n_estimators': [10,100,1000,10000]}

    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_iono[['a01',
                                                                 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09',
                                                                 'a10', 'a11', 'a12', 'a13', 'a14',
                                                                 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22',
                                                                 'a23', 'a24', 'a25', 'a26', 'a27',
                                                                 'a28', 'a29', 'a30', 'a31', 'a32', 'a33']],
                                                        df_iono['class_target'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    rf = RandomForestClassifier(n_estimators=1000, random_state=random_state)

    hp_model = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            cv=4,
                            return_train_score=True)

    hp_model.fit(X_train, y_train)

    df_hyperparameter = pd.DataFrame(hp_model.cv_results_).sort_values(by='rank_test_score')

    estimator = df_hyperparameter.iloc[0]['param_n_estimators']

    opt_params = {'n_estimators': estimator}
    print(opt_params)

    final_model = RandomForestClassifier(n_estimators=estimator, random_state=random_state)

    final_model.fit(X_train, y_train)

    preds = final_model.predict(X_test)

     # Calculate ERR
    cf = confusion_matrix(y_true=y_test, y_pred=preds)
    tn, fp, fn, tp = cf.ravel()
    ERR = (fp + fn) / (tn + fp + fn + tp)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    AUC_score = auc(fpr, tpr)

    # Calculate Logloss

    Logloss_score = log_loss(y_test, preds)

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'ERR', 'Logloss', 'AUC', 'opt_params', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                               date.minute,
                                                               date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                            ERR,
                                            Logloss_score,
                                           AUC_score,
                                            opt_params,
                                            'randomForrest']

    df_output['random_state'] = random_state

    path1 = path

    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/Iono/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)


for i in range(4):
    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/Iono"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "/{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "/{}{}{}{}".format(date.hour,date.minute,date.second,date.microsecond)

    os.mkdir(path)

    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_iono[['a01',
                                                                 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09',
                                                                 'a10', 'a11', 'a12', 'a13', 'a14',
                                                                 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22',
                                                                 'a23', 'a24', 'a25', 'a26', 'a27',
                                                                 'a28', 'a29', 'a30', 'a31', 'a32', 'a33']],
                                                        df_iono['class_target'],
                                                        test_size=0.33,
                                                        random_state=random_state)
    col_X_train = ['a01',
                   'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14',
                   'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27',
                   'a28', 'a29', 'a30', 'a31', 'a32', 'a33']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    # Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    # Create Model with GPBoost

    likelihood = 'bernoulli_probit'
    gp_model = gpb.GPModel(likelihood=likelihood, gp_coords=data_train.data)

    # Fit Gaussian Process
    gp_model.fit(y=y_train)

    preds = gp_model.predict(X_pred=X_test, gp_coords_pred=X_test, predict_var=True, predict_response=True)

    predictions = []

    for i in preds['mu']:
        if i >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate ERR
    cf = confusion_matrix(y_true=y_test, y_pred=predictions)
    tn, fp, fn, tp = cf.ravel()
    ERR = (fp + fn) / (tn + fp + fn + tp)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, preds['mu'])
    AUC_score = auc(fpr, tpr)

    # Calculate Logloss

    Logloss_score = log_loss(y_test, predictions)

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'ERR', 'Logloss','AUC', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                               date.minute,
                                                               date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           ERR,
                                           Logloss_score,
                                           AUC_score,
                                           'Gaussian Process']

    df_output['random_state'] = random_state

    path1 = path

    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/Iono/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)


