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
path_abalone = path + '/abalone.data'

#Create dict with columns
dict_columns = {0:'Sex',
                1: 'Length',
                2: 'Diameter',
                3: 'Height',
                4: 'Whole_wheigt',
                5:'Shucked_weight',
                6:'Viscera_weight',
               7:'Shell_weight',
               8: 'Number_of_Rings'}

#Load data
list_data = []
with open(path_abalone, 'r') as file:
    rows = csv.reader(file, delimiter=',')

    count = 0
    for r in rows:
        count += 1
        list_data.append(r)
print(count)

df_abalone = pd.DataFrame.from_records(list_data)

#Rename columns
df_abalone = df_abalone.rename(columns=dict_columns)

#Transform 'Sex' into numerical values
df_abalone['Sex'] = df_abalone['Sex'].astype('str')
list_sex = []
for i in df_abalone['Sex']:
    if i == 'M':
        list_sex.append(0)
    if i == 'F':
        list_sex.append(1)
    if i == 'I':
        list_sex.append(2)

df_abalone['_Sex'] = list_sex

#Change datatypes
df_abalone['Length'] = df_abalone['Length'].astype('float')
df_abalone['Diameter'] = df_abalone['Diameter'].astype('float')
df_abalone['Height'] = df_abalone['Height'].astype('float')
df_abalone['Whole_wheigt'] = df_abalone['Whole_wheigt'].astype('float')
df_abalone['Shucked_weight'] = df_abalone['Shucked_weight'].astype('float')
df_abalone['Viscera_weight'] = df_abalone['Viscera_weight'].astype('float')
df_abalone['Shell_weight'] = df_abalone['Shell_weight'].astype('float')
df_abalone['Number_of_Rings'] = df_abalone['Number_of_Rings'].astype('int')

#Create variable 'age'
df_abalone['Age'] = df_abalone['Number_of_Rings'] + 1.5

#Draw 4 samples with 500 rows
samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 4177))
df_sample_1 = df_abalone.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 4177))
df_sample_2 = df_abalone.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0,4177))
df_sample_3 = df_abalone.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 4177))
df_sample_4 = df_abalone.iloc[samples]

#Set count for data load
count = 0

#Create dict with samples
dict_samples = {'df1':df_sample_1, 'df2':df_sample_2, 'df3': df_sample_3, 'df4':df_sample_4}

#4-fold CrossValidation for GPBoost
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/Abalone"

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

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['Length',
                                                                    'Diameter',
                                                                    'Height',
                                                                    'Whole_wheigt',
                                                                    'Shucked_weight',
                                                                    'Viscera_weight',
                                                                    'Shell_weight']],
                                                        df_sample['Age'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize data
    col_X_train = ['Length',
                   'Diameter',
                   'Height',
                   'Whole_wheigt',
                   'Shucked_weight',
                   'Viscera_weight',
                   'Shell_weight']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label = y_train)

    #Create Model with GPBoost
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

    #Calculate CRPS
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

    #Set path to store output-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/Abalone/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #save model
    bst.save_model(path1 + "/model.json")

#set count for data load
count=0

#4-fold CrossValidation for Random Forest
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/Abalone"

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
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['Length',
                                                                    'Diameter',
                                                                    'Height',
                                                                    'Whole_wheigt',
                                                                    'Shucked_weight',
                                                                    'Viscera_weight',
                                                                    'Shell_weight']],
                                                        df_sample['Age'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Create RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=1000, random_state=random_state)

    #Hperparametertuning
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

    #Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/Abalone/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#Set count for data load
count=0

#4-fold CrossValidation for Gaussian Process
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/Abalone"

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

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['Length',
                                                                    'Diameter',
                                                                    'Height',
                                                                    'Whole_wheigt',
                                                                    'Shucked_weight',
                                                                    'Viscera_weight',
                                                                    'Shell_weight']],
                                                        df_sample['Age'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize data
    col_X_train = ['Length',
                   'Diameter',
                   'Height',
                   'Whole_wheigt',
                   'Shucked_weight',
                   'Viscera_weight',
                   'Shell_weight']

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

    #Make predictions
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
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/Abalone/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#Set count for data load
count = 0

#4-fold CrossValiadation for Boosting
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/Boosting/Abalone"

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

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['Length',
                                                                   'Diameter',
                                                                   'Height',
                                                                   'Whole_wheigt',
                                                                   'Shucked_weight',
                                                                   'Viscera_weight',
                                                                   'Shell_weight']],
                                                        df_sample['Age'],
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

    #Calculate CRPS
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

    #Set path to store outputs
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/Boosting/Abalone/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")