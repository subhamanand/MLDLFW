import data_pre_processing as pre
import impute_missing as m
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.externals import joblib

def regression(data, target, regression_algo, modelname):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame uploaded by user

    target: string, name of target column in data

    regression_algo: sklearn model

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    data = data.dropna(subset = [target],axis = 0)
    input_variables = list(data.columns)
    input_variables.remove(target)
    input_data = data[input_variables]
    useful_variables = pre.eliminate_redundant(input_data)
    non_missing_data = m.DataFrameImputer().fit_transform(input_data[useful_variables])
    num_cols, cat_col = pre.identify_variable(non_missing_data)
    if len(cat_col) > 0:
        # X_cat = non_missing_data[cat_col]
        # X_cat = X_cat.astype('str')
        # X_num = non_missing_data[num_cols].values
        # # labelencoder_dict, onehotencoder_dict, clean_data = pre.labelencoder(X_cat)
        # # clean_data = np.concatenate((clean_data, X_num), axis=1)
        # X_dummy = pd.get_dummies(X_cat)
        clean_data = pd.get_dummies(non_missing_data, columns=cat_col, drop_first=True, prefix_sep= '_')
    else:
        clean_data = non_missing_data
    clean_data.columns = clean_data.columns.str.replace('[^A-Za-z0-9]+', '_').str.strip('_')
    input_variables = list(clean_data.columns)
    target_data = data[target].values
    input_train, target_train, input_test, target_test = pre.train_test_split(clean_data.values, target_data)

    model = regression_algo
    model.fit(input_train, target_train)

    target_pred = model.predict(input_test)

    error = abs(target_pred - target_test)/target_test
    error[error >= 1E308] = 0
    mape =100- error.mean()*100

    joblib.dump(model, modelname)

    return useful_variables, input_variables, modelname, mape

def classification(data, target, classification_algo, modelname):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame uploaded by user

    target: string, name of target column in data

    classification_algo: sklearn classification model

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    data = data.dropna(subset = [target],axis = 0)
    input_variables = list(data.columns)
    input_variables.remove(target)
    input_data = data[input_variables]
    useful_variables = pre.eliminate_redundant(input_data)
    non_missing_data = m.DataFrameImputer().fit_transform(input_data[useful_variables])
    num_cols, cat_col = pre.identify_variable(non_missing_data)
    if len(cat_col) > 0:
        # X_cat = non_missing_data[cat_col]
        # X_cat = X_cat.astype('str')
        # X_num = non_missing_data[num_cols].values
        # # labelencoder_dict, onehotencoder_dict, clean_data = pre.labelencoder(X_cat)
        # # clean_data = np.concatenate((clean_data, X_num), axis=1)
        # X_dummy = pd.get_dummies(X_cat)
        clean_data = pd.get_dummies(non_missing_data, columns=cat_col, drop_first=True, prefix_sep= '_')
    else:
        clean_data = non_missing_data
    clean_data.columns = clean_data.columns.str.replace('[^A-Za-z0-9]+', '_').str.strip('_')
    input_variables = list(clean_data.columns)
    target_data = data[target].values
    input_train, target_train, input_test, target_test = pre.train_test_split(clean_data.values, target_data)

    model = classification_algo
    model.fit(input_train, target_train)

    joblib.dump(model, modelname)

    target_pred = model.predict(input_test)
    # print("target_pred: "+str(target_pred))
    cm = confusion_matrix(target_test, target_pred)
    acc = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    print(modelname)
    return useful_variables, input_variables, modelname, acc*100
