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

# Fix columns with comma decimals
columns_with_comma = [
    'Hoeveelheid', "Invoer bloedglucose (mmol/l)", "Eerste toediening (eenh.)",
    "Uitgebreide toediening (eenh.)", "Serienummer_bolus", "Invoer koolhydraatverbruik (g)",
    "Koolhydraatratio", "Toegediende insuline (eenh.)_bolus", "Serienummer_alarms"
]

for col in columns_with_comma:
    try:
        merged[col] = merged[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '').astype(float)
    except Exception as e:
        print(f"Fout bij conversie van kolom {col}: {e}")

# Attempt full numeric conversion just to be safe
for col in columns_with_comma:
    try:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
    except:
        pass

redundant_columns = ["Serienummer",	"Percentage (%)", "Toegediende insuline (eenh.)", "Serienummer_basal", 'Eerste toediening (eenh.)',	'Uitgebreide toediening (eenh.)', 'Serienummer_bolus', 	'Serienummer_alarms']
merged = merged.drop(columns=redundant_columns)


merged.rename(columns={
    'Tijdstempel': 'Timestemp',
    'Hoeveelheid': 'Insuline units (basal)',
    'Invoer bloedglucose (mmol/l)': 'BG_input (mmol/l)',
    'Invoer koolhydraatverbruik (g)': 'Carbohydrates (g)',
    'Koolhydraatratio': 'Carb ratio',
    'Toegediende insuline (eenh.)_bolus': 'Insuline units (bolus)',
    'Duur (minuten)': 'Duration (minutes)',
    "CGM-glucosewaarde (mmol/l)": "Glucose value (mmol/l)",
    "Alarm/Gebeurtenis": "Alarm"
}, inplace=True)

# Translate categorical values
merged["Insulinetype"] = merged["Insulinetype"].replace({
    "Gepland": "Scheduled",
    "Tijdelijk": "Temporary",
    "Onderbreken": "Suspended"
})

merged["Insulinetype_bolus"] = merged["Insulinetype_bolus"].replace({
    "Normaal": "Normal"
})


merged.to_csv("Glucose_export.csv")