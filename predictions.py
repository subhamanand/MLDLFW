import data_pre_processing as pre
import numpy as np
import pandas as pd
import impute_missing as m

def prediction(data, model, useful_variables, input_variables, fileName):
    predict_data = data[useful_variables]
    predict_data = m.DataFrameImputer().fit_transform(predict_data)
    num_col, cat_col = pre.identify_variable(predict_data)
    if len(cat_col)>0:
        X_pred = pd.get_dummies(predict_data, columns=cat_col, drop_first=True, prefix_sep= '_')
    else:
        X_pred = predict_data
    X_pred.columns = X_pred.columns.str.replace('[^A-Za-z0-9]+', '_').str.strip('_')
    if len(input_variables)>X_pred.shape[1]:
        X_pred = X_pred.reindex(columns=input_variables, fill_value=0)
    elif len(input_variables)<X_pred.shape[1]:
        excess_variables = set(X_pred.columns) - set(input_variables)
        pred_column= list(X_pred.columns)
        for e in excess_variables:
            pred_column.remove(e)
        X_pred = X_pred[pred_column]
    target_pred = model.predict(X_pred.values)
    final_table = data
    final_table['Predicted_values'] = target_pred
    print("Prediction success")
    file_path = "./output/out_"+fileName
    final_table.to_csv(file_path)
    #print(type(final_table))
    return file_path

# def logistic_regression_prediction(data, model, useful_variables, input_variables):
#     predict_data = data[useful_variables]
#     predict_data = m.DataFrameImputer().fit_transform(predict_data)
#     num_col, cat_col = pre.identify_variable(predict_data)
#     if len(cat_col)>0:
#         X_pred = pd.get_dummies(predict_data, columns=cat_col, drop_first=True, prefix_sep= '_')
#     else:
#         X_pred = predict_data
#     X_pred.columns = X_pred.columns.str.replace('[^A-Za-z0-9]+', '_').str.strip('_')
#     if len(input_variables)>X_pred.shape[1]:
#         X_pred.reindex(columns=input_variables, fill_value=0)
#     elif len(input_variables)<X_pred.shape[1]:
#         excess_variables = set(X_pred.columns) - set(input_variables)
#         pred_column= list(X_pred.columns)
#         for e in excess_variables:
#             pred_column.remove(e)
#         X_pred = X_pred[pred_column]
#     target_pred = model.predict(X_pred.values)
#     final_table = data
#     final_table['Predicted_values'] = target_pred
#     print("Prediction success")
#     return final_table

