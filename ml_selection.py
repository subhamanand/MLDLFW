import ml_algorithm as algo
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def regression_selection(data,target):
    lr_reg = LinearRegression()
    useful_variables_lr, input_variables_lr, lr_model, mape_lr = algo.regression(data, target, lr_reg, "./model/model_linear_regression.pkl")

    rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)
    useful_variables_rf, input_variables_rf, rf_model, mape_rf = algo.regression(data, target, rf_reg, "./model/model_random_forest_regression.pkl")

    if mape_rf>mape_lr:
        acc = "The model accuracy of Linear Regression is " + str(round(mape_lr, 2)) + "% #" + \
              "The model accuracy of Random Forest Regression is " + str(round(mape_rf, 2)) + "% #" + \
              "Model selected is: Random Forest"
        return rf_model, useful_variables_rf, input_variables_rf, acc
    else:
        acc = "The model accuracy of Linear Regression is " + str(round(mape_lr, 2)) + "% #" + \
              "The model accuracy of Random Forest Regression is " + str(round(mape_rf, 2)) + "% #" + \
              "Model selected is: Linear Regression"
        return lr_model, useful_variables_lr, input_variables_lr, acc


def classification_selection(data, target):
    log_reg = LogisticRegression(random_state = 0)
    useful_variables_log, input_variables_log, log_model, acc_log = algo.classification(data, target, log_reg, "./model/model_logistic_regression.pkl")


    rf_reg = RandomForestClassifier(n_estimators = 100, criterion = 'gini', random_state = 0)
    useful_variables_rf, input_variables_rf, rf_model, acc_rf = algo.classification(data, target, rf_reg, "./model/model_random_forest_classifier.pkl")

    if acc_rf > acc_log:
        acc = "Model accuracy of Logistic Regression is: " + str(round(acc_log, 2)) + "% #"+\
              "Model accuracy of Random Forest is: " + str(round(acc_rf, 2)) + "% #"+\
              "Model selected is: Random Forest"
        return rf_model, useful_variables_rf, input_variables_rf, acc
    else:
        acc = "Model accuracy of Logistic Regression is: " + str(round(acc_log, 2)) + "% #"+\
              "Model accuracy of Random Forest is: " + str(round(acc_rf, 2)) + "% #"+\
              "Model selected is: Logistic Regression"
        return log_model, useful_variables_log, input_variables_log, acc

