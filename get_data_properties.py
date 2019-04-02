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

