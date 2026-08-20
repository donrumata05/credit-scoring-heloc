from sklearn.impute import SimpleImputer

def preprocess_features(X_train, X_val, X_test):
    special_values = [-7, -8]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for col in X_train.columns:
        X_train[f'{col}_special'] = X_train[col].isin(special_values).astype(int)
        X_val[f'{col}_special'] = X_val[col].isin(special_values).astype(int)
        X_test[f'{col}_special'] = X_test[col].isin(special_values).astype(int)

    X_train = X_train.mask(X_train.isin(special_values))
    X_val = X_val.mask(X_val.isin(special_values))
    X_test = X_test.mask(X_test.isin(special_values))

    imputer = SimpleImputer(strategy='median')

    feature_names = feature_names = list(X_train.columns)

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    return X_train, X_val, X_test, imputer, feature_names