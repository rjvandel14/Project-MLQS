import numpy as np
import pandas as pd

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


#remove redundant columns
redundant = ["glucose_diff"]
df.drop(columns=redundant)


# df.to_csv("feature.csv")


