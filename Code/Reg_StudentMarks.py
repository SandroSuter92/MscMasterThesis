from sklearn.model_selection import train_test_split
import csv
import os
import gpboost as gpb
import numpy as np
import datetime
import pandas as pd
import CRPS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Set path to data folder for data load
path = 'C:/Users/localadmin/Desktop/Thesis/data'
path_sm = path + '/Student_Marks.csv'

# load data
df_sm = pd.read_csv(path_sm)

# 4-fold CrossValidation for GPBoost
for i in range(4):

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/StudentMarks"

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
    likelihood = 'gaussian'

    #Create first train params. Later they will be overwritten
    params = {'objective': likelihood,
              'learning_rate': 1,
              'max_depth': 3,
              'verbose': 0}

    if likelihood == "gaussian":
        num_boost_round = 50
        params['objective'] = 'regression_l2'
    if likelihood in ("bernoulli_probit", "bernoulli_logit"):
        num_boost_round = 500
        params['objective'] = 'binary'



    #Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [0.1,0.01], 'min_data_in_leaf': [1,10,100],  #0.1,0.01  # 10,100
                    'max_depth': [1,5,10,-1]} #5,10,-1

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sm[['number_courses', 'time_study']],
                                                        df_sm['Marks'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Normalize data
    col_X_train = ['number_courses', 'time_study']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label = y_train)

    #Create Model with GPBoost
    gp_model = gpb.GPModel(gp_coords=data_train.data, num_neighbors=30)

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
                                                 seed=1000)

    #Print statement to see best parameters
    print(opt_params)

    #Set parameters to results from above
    params = {'learning_rate': opt_params['best_params']['learning_rate'],
              'min_data_in_leaf': opt_params['best_params']['min_data_in_leaf'],
              'max_depth': opt_params['best_params']['max_depth']}

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    print("Params set: ", params)

    #Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'],
                    gp_model=gp_model)

    #Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    # Calculate root mean squared error
    RMSE = np.sqrt(np.mean((pred_resp['response_mean']- y_test) ** 2))

    # Calculate CRPS
    CRPS_score = CRPS.CRPS(pred_resp['response_mean'], y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id','Date', 'RMSE','CRPS', 'opt_params', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                            date.minute,
                                                            date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           RMSE,
                                           CRPS_score,
                                           opt_params,
                                           'Gaussian Process Boosting']

    df_output['random_state'] = random_state

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/StudentMarks/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    # save model
    bst.save_model(path1 + "/model.json")

# 4-fold CrossValidation for RandomForest
for i in range(4):

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/StudentMarks"

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
    param_grid = {'n_estimators': [10, 100, 1000, 10000]}

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sm[['number_courses', 'time_study']],
                                                        df_sm['Marks'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Create RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=1000, random_state=random_state)

    hp_model = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            cv=4,
                            return_train_score=True)

    hp_model.fit(X_train, y_train)

    df_hyperparameter = pd.DataFrame(hp_model.cv_results_).sort_values(by='rank_test_score')

    estimator = df_hyperparameter.iloc[0]['param_n_estimators']

    opt_params = {'n_estimators': estimator}
    print(opt_params)

    # Train final model
    final_model = RandomForestRegressor(n_estimators=estimator, random_state=random_state)

    final_model.fit(X_train, y_train)

    # Make Predictions
    preds = final_model.predict(X_test)

    # Calculate RMSE
    RMSE = np.sqrt(np.mean((preds - y_test) ** 2))

    # Calculate CRPS
    CRPS_score = CRPS.CRPS(preds, y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'RMSE','CRPS', 'opt_params', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MTB{}{}{}".format(date.hour,
                                                              date.minute,
                                                              date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           RMSE,
                                           CRPS_score,
                                           opt_params,
                                           'Random Forrest']

    df_output['random_state'] = random_state

    path1 = path

    # set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/StudentMarks/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

# 4-fold CrossValidation for Gaussian Process
for i in range(4):
    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/StudentMarks"

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

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sm[['number_courses', 'time_study']],
                                                        df_sm['Marks'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Normalize data
    col_X_train = ['number_courses', 'time_study']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    # Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    # Create Model with GPBoost

    likelihood = 'gaussian'
    gp_model = gpb.GPModel(likelihood=likelihood, gp_coords=data_train.data)

    # Fit Gaussian Process
    gp_model.fit(y=y_train)

    # Make prediction
    preds = gp_model.predict(X_pred=X_test, gp_coords_pred=X_test, predict_var=True, predict_response=True)

    # Calculate RMSE
    RMSE = np.sqrt(np.mean((preds['mu'] - y_test) ** 2))

    # Calcualte CRPS

    CRPS_score = CRPS.CRPS(preds['mu'], y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'RMSE', 'CRPS', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                               date.minute,
                                                               date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           RMSE,
                                           CRPS_score,
                                           'Gaussian Process']

    df_output['random_state'] = random_state

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/StudentMarks/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

# 4-fold CrossValidation for Boosting
for i in range(4):

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/Boosting/StudentMarks"

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
    likelihood = 'gaussian'

    #Create first train params. Later they will be overwritten
    params = {'objective': likelihood,
              'learning_rate': 1,
              'max_depth': 3,
              'verbose': 0}

    if likelihood == "gaussian":
        num_boost_round = 50
        params['objective'] = 'regression_l2'
    if likelihood in ("bernoulli_probit", "bernoulli_logit"):
        num_boost_round = 500
        params['objective'] = 'binary'



    #Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [0.1,0.01], 'min_data_in_leaf': [1,10,100],  #0.1,0.01  # 10,100
                    'max_depth': [1,5,10,-1]} #5,10,-1

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sm[['number_courses', 'time_study']],
                                                        df_sm['Marks'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label = y_train)

    #Hyperparametertuning
    opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                                 params=params,
                                                 num_try_random=None,
                                                 nfold=4,
                                                 train_set= data_train,
                                                 verbose_eval=1,
                                                 num_boost_round=1000,
                                                 early_stopping_rounds=10,
                                                 seed=1000)

    #Print statement to see best parameters
    print(opt_params)

    #Set parameters to results from above
    params = {'learning_rate': opt_params['best_params']['learning_rate'],
              'min_data_in_leaf': opt_params['best_params']['min_data_in_leaf'],
              'max_depth': opt_params['best_params']['max_depth']}

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    print("Params set: ", params)

    #Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'])

    #Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    # Calculate root mean squared error
    RMSE = np.sqrt(np.mean((pred_resp- y_test) ** 2))

    # Calculate CRPS
    CRPS_score = CRPS.CRPS(pred_resp, y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id','Date', 'RMSE','CRPS', 'opt_params', 'Model_type'])

    df_output.loc[len(df_output.index)] = ["MGPB{}{}{}".format(date.hour,
                                                            date.minute,
                                                            date.second),
                                           "{}_{}_{}".format(date.year,
                                                             date.month,
                                                             date.day),
                                           RMSE,
                                           CRPS_score,
                                           opt_params,
                                           'Boosting']

    df_output['random_state'] = random_state

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/Boosting/StudentMarks/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    # save model
    bst.save_model(path1 + "/model.json")