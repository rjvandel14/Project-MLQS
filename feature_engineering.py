import numpy as np
import pandas as pd
from TemporalAbstraction import CategoricalAbstraction 


df = pd.read_csv('Glucose_export.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

#Set the rolling window to 30 minutes
window = '30min'
numeric_columns = df.select_dtypes(include='number').columns
new_features = ['min', 'max', 'mean', 'std']

for col in numeric_columns:
    for feature in new_features:
        agg_glucose = df[col].rolling(window).agg([feature])
        df[feature + '_' + col] = agg_glucose[feature]

conditions = [
    df['Glucose value (mmol/l)'] < 4,
    df['Glucose value (mmol/l)'] >= 11
]
choices = ['low', 'high']
df['cat_glucose_value (mmol/l)'] = np.select(conditions, choices, default='normal')


#Determine trend, want to add strong decrease/increase?
df['glucose_diff'] = df['Glucose value (mmol/l)'].diff()
threshold = 0
df['glucose_trend'] = np.select(
    [df['glucose_diff'] > threshold,
     df['glucose_diff'] < -threshold],
    ['increasing', 'decreasing'],
    default='stable'
)
df.drop(columns=["glucose_diff"])


#Create new column with total insuling and if it only consists of basal
#need to check after imputation (now often nan because one of the two is nan)
df['Insulin_units_total'] = df['Insuline units (basal)'] + df['Insuline units (bolus)']
df['Only basal'] = 1-((df['Insuline units (bolus)'].isna()) | (df['Insuline units (bolus)'] == 0)).astype(int)

print(df.columns)
print(df.head(20))


#remove redundant columns, to be expanded
redundant = ["glucose_diff"]
df.drop(columns=redundant)


#CategoricalAbstraction
# Optionally fill NaNs with a placeholder to avoid issues during encoding
df['glucose_trend'] = df['glucose_trend'].fillna('unknown')
df['cat_glucose_value (mmol/l)'] = df['cat_glucose_value (mmol/l)'].fillna('unknown')
df['Insulinetype'] = df['Insulinetype'].fillna('unknown')
df['Insulinetype_bolus'] = df['Insulinetype_bolus'].fillna('unknown')
df['Alarm'] = df['Alarm'].fillna('unknown')

# One-hot encode selected categorical columns
categorical_columns = [
    'glucose_trend', 
    'cat_glucose_value (mmol/l)', 
    'Insulinetype', 
    'Insulinetype_bolus', 
    'Alarm'
]

df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded.to_csv("features_onehot.csv", index=False)

cat_abs = CategoricalAbstraction()

#Get binary columns used in pattern mining
cols = [col for col in df_encoded.columns if col.startswith((
    'glucose_trend_', 
    'cat_glucose_value (mmol/l)_', 
    'Insulinetype_', 
    'Insulinetype_bolus_', 
    'Alarm_'
))] + ['Only basal'] 

match = ['like'] * len(cols)
min_support = 0.2
window_size = 6
max_pattern_size = 3

abstracted_df = cat_abs.abstract_categorical(
    data_table=df_encoded.copy(),
    cols=cols,
    match=match,
    min_support=min_support,
    window_size=window_size,
    max_pattern_size=max_pattern_size
)

abstracted_df.to_csv("featiures_with_patterns.csv", index=False)

df.to_csv("feature.csv")


