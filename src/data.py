import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    return df


def validate_data(df):
    print(f'Rows: {len(df)}')
    print(f'Columns: {len(df.columns)}')
    print(f'Duplicates num: {df.duplicated().sum()}')
    print(f'NaN num: {df.isna().sum().sum()}')
    print("\nSpecial values:")
    for value in [-7, -8, -9]:
        counts = (df == value).sum()
        counts = counts[counts > 0]
        if len(counts) > 0:
            print(f"\nValue {value}:")
            print(counts)

    print(f'Target class balance: {df['RiskPerformance'].value_counts()}')
    return df

def clean_data(df):
    df = df.copy()
    features = df.drop(columns='RiskPerformance')
    df = df[~features.eq(-9).any(axis=1)].copy()
    df = df.drop_duplicates()
    df['RiskPerformance'] = df['RiskPerformance'].map({
        'Good': 0,
        'Bad': 1
    })
    return df


def analyze_special_values(df):
    for value in [-7, -8, -9]:
        print(f"\n===== {value} =====")

        for col in df.columns:
            count = (df[col] == value).sum()

            if count > 0:
                print(
                    f"{col}: {count} "
                    f"({count / len(df):.2%})"
                )
    features = df.drop(columns='RiskPerformance')

    all_minus_9 = (features == -9).all(axis=1)

    print(
        f"All features = -9: "
        f"{all_minus_9.sum()} "
        f"({all_minus_9.mean():.2%})"
    )
    for value in [-7, -8, -9]:
        print(f"\n===== {value} =====")

        mask = (df.drop(columns='RiskPerformance') == value).any(axis=1)

        print(df.loc[mask, 'RiskPerformance'].value_counts())


def split_features_target(df):
    y = df['RiskPerformance']
    X = df.drop('RiskPerformance', axis=1)
    return X, y


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        random_state=5,
        test_size=0.2,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        random_state=5,
        test_size=0.25,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_data(path):
    df = load_data(path)
    df = validate_data(df)
    analyze_special_values(df)
    df = clean_data(df)

    X, y = split_features_target(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    return X_train, X_val, X_test, y_train, y_val, y_test