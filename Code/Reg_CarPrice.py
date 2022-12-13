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

#Set path to data folder for data load
path = 'C:/Users/localadmin/Desktop/Thesis/data'
path_cp = path + '/car_price/CarPrice_Assignment.csv'

#Load data
df_cp = pd.read_csv(path_cp)

#Change various categorical variables to numeric values
fuel = df_cp['fueltype']
_fuel = []
for i in fuel:
    if i == 'gas':
        _fuel.append(1)
    else:
        _fuel.append(2)
df_cp['_fuel'] = _fuel

asp = df_cp['aspiration']
_asp = []
for i in asp:
    if i == 'std':
        _asp.append(1)
    else:
        _asp.append(2)
df_cp['_aspiration'] = _asp

dn = df_cp['doornumber']
_dn = []
for i in dn:
    if i == 'two':
        _dn.append(2)
    else:
        _dn.append(4)
df_cp['_doornumber'] = _dn

cb = df_cp['carbody']
_cb = []
for i in cb:
    if i == 'convertible':
        _cb.append(1)
    elif i == 'hatchback':
        _cb.append(2)
    elif i == 'sedan':
        _cb.append(3)
    elif i == 'wagon':
        _cb.append(4)
    elif i == 'hardtop':
        _cb.append(5)
df_cp['_carbody'] = _cb

dw = df_cp['drivewheel']
_dw = []
for i in dw:
    if i == 'rwd':
        _dw.append(1)
    elif i == 'fwd':
        _dw.append(2)
    elif i == '4wd':
        _dw.append(3)
df_cp['_drivewheel'] = _dw

el = df_cp['enginelocation']
_el = []
for i in el:
    if i == 'front':
        _el.append(1)
    else:
        _el.append(2)
df_cp['_enginelocation'] = _el

et = df_cp['enginetype']
_et = []
for i in et:
    if i == 'dohc':
        _et.append(1)
    elif i == 'ohcv':
        _et.append(2)
    elif i == 'ohc':
        _et.append(3)
    elif i == 'l':
        _et.append(4)
    elif i == 'rotor':
        _et.append(5)
    elif i == 'ohcf':
        _et.append(6)
    elif i == 'dohcv':
        _et.append(7)
df_cp['_enginetype'] = _et

cn = df_cp['cylindernumber']
_cn = []
for i in cn:
    if i == 'four':
        _cn.append(4)
    elif i == 'six':
        _cn.append(6)
    elif i == 'five':
        _cn.append(5)
    elif i == 'three':
        _cn.append(3)
    elif i == 'twelve':
        _cn.append(12)
    elif i == 'two':
        _cn.append(2)
    elif i == 'eight':
        _cn.append(8)
df_cp['_cylindernumber'] = _cn

fs = df_cp['fuelsystem']
_fs = []
for i in fs:
    if i == 'mpfi':
        _fs.append(1)
    elif i == '2bbl':
        _fs.append(2)
    elif i == 'mfi':
        _fs.append(3)
    elif i == '1bbl':
        _fs.append(4)
    elif i == 'spfi':
        _fs.append(5)
    elif i == '4bbl':
        _fs.append(6)
    elif i == 'idi':
        _fs.append(7)
    elif i == 'spdi':
        _fs.append(8)
df_cp['_fuelsystem'] = _fs

#Drop NA-Values
df_cp = df_cp.dropna()

#4-fold CrossValidation for GPBoost
for i in range(4):
    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/CarPrice"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "\{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "\{}{}{}".format(date.hour, date.minute, date.second)

    os.mkdir(path)

    # Set likelihood (if regression = "gaussian", if categorical = "bernouli_probit" or "bernouli_logit"
    likelihood = 'gaussian'

    # Create first train params. Later they will be overwritten
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

    # Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [1, 0.1, 0.01], 'min_data_in_leaf': [1, 10, 100],  # 0.1,0.01  # 10,100
                  'max_depth': [1, 5, 10, -1]}  # 5,10,-1

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_cp[['symboling', '_fuel',
                                                               '_aspiration', '_doornumber', '_carbody',
                                                               '_drivewheel', '_enginelocation', 'wheelbase',
                                                               'carlength', 'carwidth', 'carheight',
                                                               'curbweight', '_enginetype',
                                                               '_cylindernumber', 'enginesize', '_fuelsystem',
                                                               'boreratio', 'stroke',
                                                               'compressionratio', 'horsepower', 'peakrpm',
                                                               'citympg', 'highwaympg']],
                                                        df_cp['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize data
    col_X_train = ['symboling', '_fuel',
                   '_aspiration', '_doornumber', '_carbody',
                   '_drivewheel', '_enginelocation', 'wheelbase',
                   'carlength', 'carwidth', 'carheight',
                   'curbweight', '_enginetype',
                   '_cylindernumber', 'enginesize', '_fuelsystem',
                   'boreratio', 'stroke',
                   'compressionratio', 'horsepower', 'peakrpm',
                   'citympg', 'highwaympg']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    # Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    # Create Model with GPBoost
    gp_model = gpb.GPModel(gp_coords=data_train.data, num_neighbors=30)

    # Hyperparametertuning
    opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                                 params=params,
                                                 num_try_random=None,
                                                 nfold=4,
                                                 gp_model=gp_model,
                                                 use_gp_model_for_validation=True,
                                                 train_set=data_train,
                                                 verbose_eval=1,
                                                 num_boost_round=1000,
                                                 early_stopping_rounds=10,
                                                 seed=1000)

    # Print statement to see best parameters
    print(opt_params)

    # Set parameters to results from above
    params = {'learning_rate': opt_params['best_params']['learning_rate'],
              'min_data_in_leaf': opt_params['best_params']['min_data_in_leaf'],
              'max_depth': opt_params['best_params']['max_depth']}

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    print("Params set: ", params)

    # Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'],
                    gp_model=gp_model)

    # Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    # Calculate root mean squared error
    RMSE = np.sqrt(np.mean((pred_resp['response_mean'] - y_test) ** 2))

    #Calculate CRPS
    CRPS_score = CRPS.CRPS(pred_resp['response_mean'], y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'RMSE', 'CRPS', 'opt_params', 'Model_type'])

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

    #Set path to store output-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/CarPrice/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")

#4-fold CrossValidation for Random Forest
for i in range(4):
    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/CarPrice"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "/{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "/{}{}{}{}".format(date.hour, date.minute, date.second, date.microsecond)

    os.mkdir(path)

    # Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'n_estimators': [10, 100, 1000, 10000]}

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_cp[['symboling', '_fuel',
                                                               '_aspiration', '_doornumber', '_carbody',
                                                               '_drivewheel', '_enginelocation', 'wheelbase',
                                                               'carlength', 'carwidth', 'carheight',
                                                               'curbweight', '_enginetype',
                                                               '_cylindernumber', 'enginesize', '_fuelsystem',
                                                               'boreratio', 'stroke',
                                                               'compressionratio', 'horsepower', 'peakrpm',
                                                               'citympg', 'highwaympg']],
                                                        df_cp['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Creat RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=1000, random_state=random_state)

    #Hyperparametertuning
    hp_model = GridSearchCV(estimator=rf,
                            param_grid=param_grid,
                            cv=4,
                            return_train_score=True)

    hp_model.fit(X_train, y_train)

    df_hyperparameter = pd.DataFrame(hp_model.cv_results_).sort_values(by='rank_test_score')

    estimator = df_hyperparameter.iloc[0]['param_n_estimators']

    opt_params = {'n_estimators': estimator}
    print(opt_params)

    #Train final model
    final_model = RandomForestRegressor(n_estimators=estimator, random_state=random_state)

    final_model.fit(X_train, y_train)

    #Make predictions
    preds = final_model.predict(X_test)

    #Calculate RMSE
    RMSE = np.sqrt(np.mean((preds - y_test) ** 2))

    #Calculate CRPS
    CRPS_score = CRPS.CRPS(preds, y_test.mean()).compute()[0]

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'RMSE', 'CRPS', 'opt_params', 'Model_type'])

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

    #Set path to store output-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/CarPrice/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#4-fold CrossValidation for Gaussian Process
for i in range(4):
    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/CarPrice"

    if os.path.isdir(path) == True:
        path = path

    else:
        os.mkdir(path)

    path = path + "/{}_{}_{}".format(date.year, date.month, date.day)

    if os.path.isdir(path) == True:
        path = path
    else:
        os.mkdir(path)

    path = path + "/{}{}{}{}".format(date.hour, date.minute, date.second, date.microsecond)

    os.mkdir(path)

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_cp[['symboling','_fuel',
                                                               '_aspiration', '_doornumber', '_carbody',
                                                               '_drivewheel', '_enginelocation', 'wheelbase',
                                                               'carlength', 'carwidth', 'carheight',
                                                               'curbweight', '_enginetype',
                                                               '_cylindernumber', 'enginesize', '_fuelsystem',
                                                               'boreratio', 'stroke',
                                                               'compressionratio', 'horsepower', 'peakrpm',
                                                               'citympg', 'highwaympg']],
                                                        df_cp['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize data
    col_X_train = ['symboling', '_fuel',
                   '_aspiration', '_doornumber', '_carbody',
                   '_drivewheel', '_enginelocation', 'wheelbase',
                   'carlength', 'carwidth', 'carheight',
                   'curbweight', '_enginetype',
                   '_cylindernumber', 'enginesize', '_fuelsystem',
                   'boreratio', 'stroke',
                   'compressionratio', 'horsepower', 'peakrpm',
                   'citympg', 'highwaympg']

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

    #Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/CarPrice/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#4-fold CrossValidation for Boosting
for i in range(4):
    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/Boosting/CarPrice"

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

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_cp[['symboling', '_fuel',
                                                               '_aspiration', '_doornumber', '_carbody',
                                                               '_drivewheel', '_enginelocation', 'wheelbase',
                                                               'carlength', 'carwidth', 'carheight',
                                                               'curbweight', '_enginetype',
                                                               '_cylindernumber', 'enginesize', '_fuelsystem',
                                                               'boreratio', 'stroke',
                                                               'compressionratio', 'horsepower', 'peakrpm',
                                                               'citympg', 'highwaympg']],
                                                        df_cp['price'],
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

    #Calculate CRPS
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

    #Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/Boosting/CarPrice/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #save model
    bst.save_model(path1 + "/model.json")