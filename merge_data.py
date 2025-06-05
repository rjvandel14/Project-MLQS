import pandas as pd

basal = pd.read_csv("data/Insulin data/basal_data_1.csv", skiprows=1)
bolus = pd.read_csv("data/Insulin data/bolus_data_1.csv", skiprows=1)
alarms = pd.read_csv("data/alarms_data_1.csv", skiprows=1)
cgm1 = pd.read_csv("data/cgm_data_1.csv", skiprows=1)
cgm2 = pd.read_csv("data/cgm_data_2.csv", skiprows=1)

cgm = pd.concat([cgm1, cgm2])

for df in [basal, bolus, cgm, alarms]:
    df["Tijdstempel"] = pd.to_datetime(df["Tijdstempel"], dayfirst=True, errors="coerce")
    df.set_index("Tijdstempel", inplace=True)
    df.sort_index(inplace=True)

# Define resampling function
def resample_mixed(df):
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    numeric_resampled = numeric.resample("5min").mean() if not numeric.empty else pd.DataFrame(index=df.resample("5min").mean().index)
    categorical_resampled = categorical.resample("5min").first() if not categorical.empty else pd.DataFrame(index=df.resample("5min").mean().index)

    return pd.concat([numeric_resampled, categorical_resampled], axis=1)

basal_resampled = resample_mixed(basal)
bolus_resampled = resample_mixed(bolus)
cgm_resampled = resample_mixed(cgm)
alarms_resampled = resample_mixed(alarms)

merged = cgm_resampled \
    .join(basal_resampled, rsuffix="_basal") \
    .join(bolus_resampled, rsuffix="_bolus") \
    .join(alarms_resampled, rsuffix="_alarms")


merged.to_csv("Glucose_export.csv")