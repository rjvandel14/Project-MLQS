import numpy as np
import pandas as pd
from TemporalAbstraction import CategoricalAbstraction 

def feature_engineering(df, min_support, window_size, max_pattern_size):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    #Set the rolling window to 30 minutes checken
    window = '30min'
    numeric_columns = df.select_dtypes(include='number').columns
    new_features = ['min', 'max', 'mean', "median",  'std']

    for col in numeric_columns:
        for feature in new_features:
            agg_glucose = df[col].rolling(window).agg([feature])
            df[feature + '_' + col] = agg_glucose[feature]

    #categorize GCM values
    conditions = [
        df['Glucose value (mmol/l)'] < 3,
        (df['Glucose value (mmol/l)'] >= 3) & (df['Glucose value (mmol/l)'] <= 3.8),
        (df['Glucose value (mmol/l)'] >= 3.9) & (df['Glucose value (mmol/l)'] <= 10),
        (df['Glucose value (mmol/l)'] > 10) & (df['Glucose value (mmol/l)'] <= 13.9),
        df['Glucose value (mmol/l)'] > 13.9
    ]
    choices = [0, 1, 2, 3, 4]
    df['cat_glucose_value (mmol/l)'] = np.select(conditions, choices)

    #Calculate trend between t and t-3
    df['glucose_diff'] = df['Glucose value (mmol/l)'] - df['Glucose value (mmol/l)'].shift(3)
    df['glucose_trend'] = np.select(
        [df['glucose_diff'] > 0,
        df['glucose_diff'] == 0,
        df['glucose_diff'] < 0,
        ],
        ['increasing', 'stable', 'decreasing']
    )

    #Create new column with total insuling and if it only consists of basal --> more interesting to do 1 IF both present?
    df['Insulin_units_total'] = df['Insuline units (basal)'] + df['Insuline units (bolus)']
    df['Only basal'] = ((df['Insuline units (bolus)'].isna()) | (df['Insuline units (bolus)'] == 0)).astype(int)

    #Day of week feature
    df['day_of_week'] = df.index.dayofweek
    df.to_csv("feature.csv")


    # One-hot encode selected categorical columns
    categorical_columns = [
        'glucose_trend', 
        'cat_glucose_value (mmol/l)', 
        'Insulinetype', 
        'Insulinetype_bolus', 
        'Alarm'
    ]

    for col in categorical_columns:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummies], axis=1)
    
    df.to_csv("features_onehot.csv", index=False)

    cat_abs = CategoricalAbstraction()

    #Get binary columns used in pattern mining
    cols = [col for col in df.columns if (
        col.startswith(('glucose_trend_', 'cat_glucose_value (mmol/l)_', 
                        'Insulinetype_', 'Insulinetype_bolus_', 'Alarm_')) 
        and df[col].dropna().isin([0, 1]).all()
    )]
    cols += ['Only basal']

    match = ['like'] * len(cols)
    min_support = 0.2
    window_size = 3
    max_pattern_size = 2

    abstracted_df = cat_abs.abstract_categorical(
        data_table=df.copy(),
        cols=cols,
        match=match,
        min_support=min_support,
        window_size=window_size,
        max_pattern_size=max_pattern_size
    )

    return abstracted_df



df_train = pd.read_csv("Glucose_export_imputed.csv")
df_test = pd.read_csv("Glucose_export_imputed_test.csv")

# Add a column to distinguish between train and test
df_train['__set__'] = 'train'
df_test['__set__'] = 'test'

# Append the two datasets
df_combined = pd.concat([df_train, df_test], ignore_index=True)


df_combined_features = feature_engineering(df_combined, 0.2,3,3)
df_train_features = df_combined_features[df_combined_features['__set__'] == 'train'].drop(columns='__set__')
df_test_features = df_combined_features[df_combined_features['__set__'] == 'test'].drop(columns='__set__')

df_train_features.to_csv("all_features_train.csv")
df_test_features.to_csv("all_features_test.csv")
