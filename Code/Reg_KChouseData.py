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
path_kc = path + '/kc_house_data.csv'

#Load data
df_kc = pd.read_csv(path_kc)

#Draw 4 random samples with 500 rows
samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 21612))
df_sample_1 = df_kc.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 21612))
df_sample_2 = df_kc.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0,21612))
df_sample_3 = df_kc.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 21612))
df_sample_4 = df_kc.iloc[samples]

#Set count for data load
count = 0

# Create dict with 4 samples
dict_samples = {'df1':df_sample_1, 'df2':df_sample_2, 'df3': df_sample_3, 'df4':df_sample_4}

#4-fold CrossValidation for GPBoost
for i in range(4):

    # load data
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/kcHouse"

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
              'learning_rate': 0.01,
              'max_depth': 3,
              'verbose': 0}

    if likelihood == "gaussian":
        num_boost_round = 50
        params['objective'] = 'regression_l2'
    if likelihood in ("bernoulli_probit", "bernoulli_logit"):
        num_boost_round = 500
        params['objective'] = 'binary'

    #Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [1,0.1,0.01], 'min_data_in_leaf': [1,10,100],  #0.1,0.01  # 10,100
                    'max_depth': [1,5,10,-1]} #5,10,-1

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['bedrooms', 'bathrooms', 'sqft_living',
                                                               'sqft_lot', 'floors', 'waterfront',
                                                               'view', 'condition', 'grade', 'sqft_above',
                                                               'sqft_basement', 'yr_built', 'yr_renovated',
                                                               'zipcode', 'lat', 'long', 'sqft_living15',
                                                               'sqft_lot15']],
                                                        df_sample['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Normalize data
    col_X_train = ['bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront',
                   'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated',
                   'zipcode', 'lat', 'long', 'sqft_living15',
                   'sqft_lot15']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label = y_train)

    #Create Model with GPBoost, due to long runtimes -> Veccia Approximation
    gp_model = gpb.GPModel(gp_coords=data_train.data)
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
    df_output['sample_seed'] = sample_seed

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/kcHouse/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")

# Set count for data load
count=0

#4-fold CrossValidation for Random Forest
for i in range(4):

    # load data
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/kcHouse"

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

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['bedrooms', 'bathrooms', 'sqft_living',
                                                                   'sqft_lot', 'floors', 'waterfront',
                                                                   'view', 'condition', 'grade', 'sqft_above',
                                                                   'sqft_basement', 'yr_built', 'yr_renovated',
                                                                   'zipcode', 'lat', 'long', 'sqft_living15',
                                                                   'sqft_lot15']],
                                                        df_sample['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Create RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=1000, random_state=random_state)

    # Hyperparametertuning
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
    df_output['sample_seed'] = sample_seed

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/kcHouse/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

# Set count for data load
count=0

# 4-fold CrossValidation for Gaussian Process
for i in range(4):

    # load data
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/kcHouse"

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
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['bedrooms', 'bathrooms', 'sqft_living',
                                                                   'sqft_lot', 'floors', 'waterfront',
                                                                   'view', 'condition', 'grade', 'sqft_above',
                                                                   'sqft_basement', 'yr_built', 'yr_renovated',
                                                                   'zipcode', 'lat', 'long', 'sqft_living15',
                                                                   'sqft_lot15']],
                                                        df_sample['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Normalize data
    col_X_train = ['bedrooms', 'bathrooms', 'sqft_living',
                   'sqft_lot', 'floors', 'waterfront',
                   'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated',
                   'zipcode', 'lat', 'long', 'sqft_living15',
                   'sqft_lot15']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    # Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    # Create Model with GPBoost

    likelihood = 'gaussian'
    gp_model = gpb.GPModel(likelihood=likelihood, gp_coords=data_train.data)
    gp_model.set_optim_params({'optimizer_cov': 'nelder_mead'})

    # Fit Gaussian Process
    gp_model.fit(y=y_train)

    # Make predictions
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
    df_output['sample_seed'] = sample_seed

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/kcHouse/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#Set count for data load
count = 0

# 4-fold CrossValidation for Boosting
for i in range(4):

    # load data
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/Boosting/kcHouse"

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
              'learning_rate': 1,
              'max_depth': 3,
              'verbose': 0}

    if likelihood == "gaussian":
        num_boost_round = 50
        params['objective'] = 'regression_l2'
    if likelihood in ("bernoulli_probit", "bernoulli_logit"):
        num_boost_round = 500
        params['objective'] = 'binary'

    # Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [0.1, 0.01], 'min_data_in_leaf': [1, 10, 100],  # 0.1,0.01  # 10,100
                  'max_depth': [1, 5, 10, -1]}  # 5,10,-1

    # Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['bedrooms', 'bathrooms', 'sqft_living',
                                                                   'sqft_lot', 'floors', 'waterfront',
                                                                   'view', 'condition', 'grade', 'sqft_above',
                                                                   'sqft_basement', 'yr_built', 'yr_renovated',
                                                                   'zipcode', 'lat', 'long', 'sqft_living15',
                                                                   'sqft_lot15']],
                                                        df_sample['price'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    # Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    # Hyperparametertuning
    opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid,
                                                 params=params,
                                                 num_try_random=None,
                                                 nfold=4,
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
                    num_boost_round=opt_params['best_iter'])

    # Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    # Calculate root mean squared error
    RMSE = np.sqrt(np.mean((pred_resp - y_test) ** 2))

    # Calculate CRPS
    CRPS_score = CRPS.CRPS(pred_resp, y_test.mean()).compute()[0]

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
                                           'Boosting']

    df_output['random_state'] = random_state

    path1 = path

    # Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/Boosting/kcHouse/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")