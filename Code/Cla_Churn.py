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

#Set path to data folder for data load
path = 'C:/Users/localadmin/Desktop/Thesis/data'
path_churn = path + '/Churn_Modelling.csv'

#Load data
df_churn = pd.read_csv(path_churn)

#Clean data (delete rows and rename the remaining)
col_not_used = ['RowNumber', 'CustomerId', 'Surname']
col_to_change = ['Geography', 'Gender']

df_churn = df_churn.drop(col_not_used, axis=1)

for col in col_to_change:
    list_values = df_churn[col]
    if col == 'Geography':
        new_list = []
        for value in list_values:
            if value == 'France':
                new_list.append(0)
            elif value == 'Spain':
                new_list.append(1)
            else:
                new_list.append(2)
        df_churn['_Geography'] = new_list
    else:
        new_list = []
        for value in list_values:
            if value == 'Female':
                new_list.append(0)
            else:
                new_list.append(1)
        df_churn['_Gender'] = new_list

#Drop NA-values
df_churn = df_churn.dropna()

#Create 4 samples of 500 rows
samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 7900))
df_sample_1 = df_churn.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 7900))
df_sample_2 = df_churn.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 7900))
df_sample_3 = df_churn.iloc[samples]

samples = []
sample_seed = np.random.randint(0, 100)
np.random.seed(sample_seed)
for i in range(500):
    samples.append(np.random.randint(0, 7900))
df_sample_4 = df_churn.iloc[samples]

#Set count for data load
count = 0

#Create dict with 4 data samples
dict_samples = {'df1':df_sample_1, 'df2':df_sample_2, 'df3': df_sample_3, 'df4':df_sample_4}

#4-fold CrossValidation for GPBoost
for i in range(4):

    #Data Load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GPB/Churn"

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
    param_grid = {'learning_rate': [1,0.1,0.01], 'min_data_in_leaf': [1,10,100],  #0.1,0.01  # 10,100
                    'max_depth': [1,5,10,-1]} #5,10,-1

    #Train-Test Split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['CreditScore', 'Age', 'Tenure',
                                                                     'Balance','NumOfProducts', 'HasCrCard',
                                                                     'IsActiveMember', 'EstimatedSalary',
                                                                     '_Geography', '_Gender']],
                                                        df_sample['Exited'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize Data
    col_X_train = ['CreditScore', 'Age', 'Tenure',
                   'Balance','NumOfProducts', 'HasCrCard',
                   'IsActiveMember', 'EstimatedSalary',
                   '_Geography', '_Gender']

    for col in col_X_train:
        X_train[col] = (X_train[col] - X_train[col].mean()) / np.std(X_train[col])
        X_test[col] = (X_test[col] - X_test[col].mean()) / np.std(X_test[col])

    #Create DataSet
    data_train = gpb.Dataset(data=X_train,
                             label=y_train)

    #Create Model with GPBoost
    gp_model = gpb.GPModel(likelihood=likelihood,gp_coords=data_train.data)

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

    print("Params set: ", params)

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    #Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'],
                    gp_model=gp_model)

    #Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    #Transform predictions into 0 and 1 values
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

    Logloss_score = log_loss(y_test,pred_resp['response_mean'])

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
    df_output['sample_seed'] = sample_seed

    path1 = path

    #Set path to store outputfile
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GPB/Churn/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")

# Set count for data load
count = 0
#4-fold CrossValidation for Random Forest
for i in range(4):

    #Set count for data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/randomForrest/Churn"

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
    param_grid = {'n_estimators': [10,100,1000,10000]}  # 5,10,-1

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['CreditScore', 'Age', 'Tenure',
                                                                   'Balance', 'NumOfProducts', 'HasCrCard',
                                                                   'IsActiveMember', 'EstimatedSalary',
                                                                   '_Geography', '_Gender']],
                                                        df_sample['Exited'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Create RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=1000, random_state=random_state)

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

    #Train final random forest model
    final_model = RandomForestClassifier(n_estimators=estimator, random_state=random_state)

    final_model.fit(X_train, y_train)

    #Make predictions
    preds = final_model.predict_proba(X_test)

    #Transform predictions into 0 and 1 values
    predictions = []

    preds_1 = []
    for i in preds:
        preds_1.append(i[1])

        if i[0] >= 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    # Calculate ERR
    cf = confusion_matrix(y_true=y_test, y_pred=predictions)
    tn, fp, fn, tp = cf.ravel()
    ERR = (fp + fn) / (tn + fp + fn + tp)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, preds_1)
    AUC_score = auc(fpr, tpr)

    # Calculate Logloss

    Logloss_score = log_loss(y_test, preds_1)

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
    df_output['sample_seed'] = sample_seed

    path1 = path

    #Set path to store outout-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/randomForrest/Churn/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#Set count for data load
count = 0

#4-fold CrossValidation for Gaussian Process
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    #Create time variable to make some ID's
    date = datetime.datetime.now()

    #Check wether path is existing or not & creating needed folders, if not existing
    path = "models/GP/Churn"

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
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['CreditScore', 'Age', 'Tenure',
                                                                   'Balance', 'NumOfProducts', 'HasCrCard',
                                                                   'IsActiveMember', 'EstimatedSalary',
                                                                   '_Geography', '_Gender']],
                                                        df_sample['Exited'],
                                                        test_size=0.33,
                                                        random_state=random_state)

    #Normalize data
    col_X_train = ['CreditScore', 'Age', 'Tenure',
                   'Balance', 'NumOfProducts', 'HasCrCard',
                   'IsActiveMember', 'EstimatedSalary',
                   '_Geography', '_Gender']

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

    #Make predictions
    preds = gp_model.predict(X_pred=X_test, gp_coords_pred=X_test, predict_var=True, predict_response=True)

    #Transform predictions into 0 and 1 values
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

    Logloss_score = log_loss(y_test, preds['mu'])

    # Create Dataframe with outputs
    df_output = pd.DataFrame(columns=['Model_Id', 'Date', 'ERR','AUC', 'Logloss', 'Model_type'])

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
    df_output['sample_seed'] = sample_seed

    path1 = path

    #Set path to store output-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/GP/Churn/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

#Set count for data load
count = 0
#4-fold CrossValidation for Boosting
for i in range(4):

    #Data load
    count += 1
    df_sample = dict_samples['df{}'.format(count)]

    # Create time variable to make some ID's
    date = datetime.datetime.now()

    # Check wether path is existing or not & creating needed folders, if not existing
    path = "models/Boosting/Churn"

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
    likelihood = 'bernoulli_probit'

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

    print('First params: ', params)

    # Param_grid for hyperparameter tuning (These grid results in 45 combinations
    param_grid = {'learning_rate': [1, 0.1, 0.01], 'min_data_in_leaf': [1, 10, 100],  # 0.1,0.01  # 10,100
                  'max_depth': [1, 5, 10, -1]}  # 5,10,-1

    #Train Test split
    random_state = np.random.randint(1000000)
    X_train, X_test, y_train, y_test = train_test_split(df_sample[['CreditScore', 'Age', 'Tenure',
                                                                   'Balance', 'NumOfProducts', 'HasCrCard',
                                                                   'IsActiveMember', 'EstimatedSalary',
                                                                   '_Geography', '_Gender']],
                                                        df_sample['Exited'],
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

    print("Params set: ", params)

    if opt_params['best_iter'] == 0:
        opt_params['best_iter'] = 1

    # Train Model
    bst = gpb.train(params=params,
                    train_set=data_train,
                    num_boost_round=opt_params['best_iter'])

    # Make predictions
    pred_resp = bst.predict(data=X_test,
                            gp_coords_pred=X_test,
                            pred_latent=False)

    #Transform predictions into 0 and 1 values
    predictions = []

    for i in pred_resp:
        if i >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    # Calculate ERR
    cf = confusion_matrix(y_true=y_test, y_pred=predictions)
    tn, fp, fn, tp = cf.ravel()
    ERR = (fp + fn) / (tn + fp + fn + tp)

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(y_test, pred_resp)
    AUC_score = auc(fpr, tpr)

    # Calculate Logloss

    Logloss_score = log_loss(y_test, pred_resp)

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
                                           'Boosting']

    df_output['random_state'] = random_state

    path1 = path

    #Set path to store output-file
    path = "C:/Users/localadmin/Desktop/Thesis/Code/models/Boosting/Churn/outputs.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df.loc[len(df.index)] = df_output.iloc[0]
        df.to_csv(path, index=False)

    else:
        df_output.to_csv(path, index=False)

    #Save model
    bst.save_model(path1 + "/model.json")

