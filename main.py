import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_classif
from xgboost import XGBRegressor, XGBClassifier
import argparse
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def preprocess_data(df, is_train=True, target_col=None):
    df = df.copy()
    df = df.dropna(how='all')

    # Remove columns
    useless_cols = ['nearest_train_station', 'highlights_attractions',
                    'ideal_for', 'traffic', 'suburbpopulation']
    df = df.drop(columns=[col for col in useless_cols if col in df.columns])

    # Better property size feature
    df['property_size_log'] = np.log1p(df['property_size'])
    df['large_property'] = (df['property_size'] >
                            df['property_size'].median()).astype(int)

    # Enhanced date features
    if 'date_sold' in df.columns:
        df['date_sold'] = pd.to_datetime(df['date_sold'], errors='coerce')
        df['year_sold'] = df['date_sold'].dt.year
        df['month_sold'] = df['date_sold'].dt.month
        df['day_of_week'] = df['date_sold'].dt.dayofweek
        df['is_weekend'] = (df['date_sold'].dt.dayofweek >= 5).astype(int)
        df['quarter_sold'] = df['date_sold'].dt.quarter
        df['days_since_2000'] = (
            df['date_sold'] - pd.Timestamp('2000-01-01')).dt.days
        df = df.drop('date_sold', axis=1)

    # Handle categoricals
    cat_cols = ['suburb', 'region', 'ethnic_breakdown']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    if is_train and target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        return X, y
    return df


def create_preprocessor(X):
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor


def main(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    test_df_eval = test_df.dropna(how='all')

    # Part I: Regression
    X_train_reg, y_train_reg = preprocess_data(train_df, target_col='price')
    X_test_reg, y_test_reg = preprocess_data(test_df_eval, target_col='price')

    preprocessor_reg = create_preprocessor(X_train_reg)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_reg, y_train_reg, test_size=0.2, random_state=42)

    preprocessor_reg.fit(X_train)
    X_train_trans = preprocessor_reg.transform(X_train)
    X_val_trans = preprocessor_reg.transform(X_val)

    # Optimized XGBoost model
    reg = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1,
        reg_lambda=80,
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'
    )

    print("Training regression model...")
    reg.fit(
        X_train_trans, y_train,
        eval_set=[(X_val_trans, y_val)],
        early_stopping_rounds=50,
        verbose=10
    )

    # Create full pipeline for final model
    reg_model = Pipeline(steps=[
        ('preprocessor', preprocessor_reg),
        ('feature_selection', SelectKBest(
            score_func=mutual_info_regression, k='all')),
        ('regressor', reg)
    ])

    # Final training on full data
    print("\nFinal training on full dataset...")
    reg_model.fit(X_train_reg, y_train_reg)

    # Cross-validation
    print("\nRunning cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        reg_model,
        X_train_reg,
        y_train_reg,
        cv=kfold,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    print("\nCross-Validation MAE Scores:")
    print([f"{-x:,.0f}" for x in cv_scores])
    print(f"Mean CV MAE: {-cv_scores.mean():,.0f} (±{-cv_scores.std():,.0f})")

    # Evaluation
    train_predictions = reg_model.predict(X_train_reg)
    test_predictions = reg_model.predict(X_test_reg)
    train_mae = mean_absolute_error(y_train_reg, train_predictions)
    test_mae = mean_absolute_error(y_test_reg, test_predictions)
    print(f"\nTrain MAE: {train_mae:,.0f}")
    print(f"Test MAE: {test_mae:,.0f}")
    print(f"Train-Test Gap: {test_mae - train_mae:,.0f}")
    print("---------------------------")

    reg_results = pd.DataFrame({
        'id': test_df['id'],
        'price': reg_model.predict(preprocess_data(test_df, is_train=False))
    })
    reg_results['price'] = reg_results['price'].map(lambda x: f"{x:.1f}")
    reg_results.to_csv('main.regression.csv', index=False)

    # Part II: Classification
    X_train_clf, y_train_clf = preprocess_data(train_df, target_col='type')
    X_test_clf, y_test_clf = preprocess_data(test_df_eval, target_col='type')

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_train_clf_encoded = label_encoder.fit_transform(y_train_clf)

    # Create and fit classification model
    preprocessor_clf = create_preprocessor(X_train_clf)
    clf_model = Pipeline(steps=[
        ('preprocessor', preprocessor_clf),
        ('feature_selection', SelectKBest(score_func=f_classif, k=14)),
        ('classifier', XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.02,
            subsample=1,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=10,
            random_state=42
        ))
    ])

    clf_model.fit(X_train_clf, y_train_clf_encoded)

    # Predict and inverse-transform labels
    train_clf_predictions = label_encoder.inverse_transform(
        clf_model.predict(X_train_clf))
    test_clf_predictions = label_encoder.inverse_transform(
        clf_model.predict(X_test_clf))

    # Calculate F1 scores
    train_f1 = f1_score(y_train_clf, train_clf_predictions,
                        average='weighted', zero_division=1)
    test_f1 = f1_score(y_test_clf, test_clf_predictions,
                       average='weighted', zero_division=1)

    print(f"Train F1: {train_f1:.3f}")
    print(f"Test F1: {test_f1:.3f}")

    X_test_clf_output = preprocess_data(test_df_eval, is_train=False)
    clf_predictions_output = clf_model.predict(X_test_clf_output)
    clf_predictions_output_labels = label_encoder.inverse_transform(
        clf_predictions_output)

    clf_results = pd.DataFrame({
        'id': test_df['id'],
        'type': clf_predictions_output_labels
    })
    clf_results.to_csv('main.classification.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', type=str)
    parser.add_argument('test_file', type=str)
    args = parser.parse_args()
    main(args.train_file, args.test_file)
