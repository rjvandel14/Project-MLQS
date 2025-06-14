import pandas as pd
from DomainSpecificImputation import DomainSpecificImputation, find_best_glucose_window

# Load train data
df = pd.read_csv("result_outliers.csv", parse_dates=["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Impute basal insulin
imputer = DomainSpecificImputation()

#best_window, error_dict = find_best_glucose_window(df.copy(), imputer)

df = imputer.impute_glucose_sliding_dynamic(df, max_window_minutes=60)
df = imputer.impute_basal_insulin(df)
df = imputer.impute_bolus_insulin(df)
df = imputer.impute_alarms(df)

# Save result
df.to_csv("Glucose_export_imputed.csv")

# Load test data
df = pd.read_csv("result_outliers_test.csv", parse_dates=["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Impute basal insulin
imputer = DomainSpecificImputation()

#best_window, error_dict = find_best_glucose_window(df.copy(), imputer)

df = imputer.impute_glucose_sliding_dynamic(df, max_window_minutes=60)
df = imputer.impute_basal_insulin(df)
df = imputer.impute_bolus_insulin(df)
df = imputer.impute_alarms(df)

# Save result
df.to_csv("Glucose_export_imputed_test.csv")
