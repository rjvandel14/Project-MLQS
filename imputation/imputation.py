import pandas as pd
from DomainSpecificImputation import DomainSpecificImputation

# Load data
df = pd.read_csv("Glucose_export.csv", parse_dates=["Timestamp"])
df.set_index("Timestamp", inplace=True)

# Impute basal insulin
imputer = DomainSpecificImputation()
df = imputer.impute_glucose_sliding(df, window_minutes=20)
df = imputer.impute_basal_insulin(df)
df = imputer.impute_bolus_insulin(df)
df = imputer.impute_alarms(df)

# Save result
df.to_csv("Glucose_export_basal_imputed.csv")
