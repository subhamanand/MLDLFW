from impute_missing import DataFrameImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd


def eliminate_redundant(dataset):
    """ Eliminate variables with more than 50% missing values or more than 10 distinct categories"""

    cols = list(dataset.columns)
    data = dataset[cols]
    num_cols, cat_cols = identify_variable(data)
    # if len(num_cols)>0 and len(cat_cols)>0:
    size = len(data)
    missing_data = data.isna()
    useful_columns = list(missing_data.columns)
    for cols in useful_columns:
        if missing_data[cols].sum()/size >= 0.5:
            useful_columns.remove(cols)
        elif cols in cat_cols and dataset[cols].nunique() > 10:
            useful_columns.remove(cols)
    return useful_columns

def labelencoder(X):
    labelencoder_dict = {}
    onehotencoder_dict = {}
    X_train = None
    for i in range(0, X.shape[1]):
        label_encoder = LabelEncoder()
        labelencoder_dict[i] = label_encoder
        feature = label_encoder.fit_transform(X[:, i])
        feature = feature.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        onehotencoder_dict[i] = onehot_encoder
        if X_train is None:
            X_train = feature
        else:
            X_train = np.concatenate((X_train, feature), axis=1)
    return labelencoder_dict, onehotencoder_dict, X_train

def identify_variable(data):
    cols = list(data.columns)
    dataset = data[cols]
    num_cols = list(dataset._get_numeric_data().columns)
    cat_cols = list(set(cols) - set(num_cols))
    return num_cols, cat_cols

def train_test_split(input,target):
    from sklearn.model_selection import train_test_split
    input_train, input_test, target_train, target_test = train_test_split(input, target, test_size=0.25, random_state=0)
    return input_train, target_train, input_test, target_test

def standardize(input_data,con_cols):
    """ standardize columns """
    data_scaler = input_data
    for col in con_cols:
        data_scaler[col] = (data_scaler[col]-data_scaler[col].mean())/data_scaler[col].std()
    return data_scaler

def clean_column_names(data):
    data.columns = data.columns.str.replace('[^A-Za-z0-9]+', '_').str.strip('_')
    return data

def getEncoded(test_data,labelencoder_dict,onehotencoder_dict):
    test_encoded_x = None
    for i in range(0,test_data.shape[1]):
        label_encoder =  labelencoder_dict[i]
        feature = label_encoder.transform(test_data[:,i])
        feature = feature.reshape(test_data.shape[0], 1)
        onehot_encoder = onehotencoder_dict[i]
        feature = onehot_encoder.transform(feature)
        if test_encoded_x is None:
            test_encoded_x = feature
        else:
            test_encoded_x = np.concatenate((test_encoded_x, feature), axis=1)
    return test_encoded_x