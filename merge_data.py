# import pandas as pd

# basal = pd.read_csv("data/Insulin data/basal_data_1.csv", skiprows=1)
# bolus = pd.read_csv("data/Insulin data/bolus_data_1.csv", skiprows=1)
# alarms = pd.read_csv("data/alarms_data_1.csv", skiprows=1)
# cgm1 = pd.read_csv("data/cgm_data_1.csv", skiprows=1)
# cgm2 = pd.read_csv("data/cgm_data_2.csv", skiprows=1)

# cgm = pd.concat([cgm1, cgm2])

# for df in [basal, bolus, cgm, alarms]:
#     df["Tijdstempel"] = pd.to_datetime(df["Tijdstempel"], dayfirst=True, errors="coerce")
#     df.set_index("Tijdstempel", inplace=True)
#     df.sort_index(inplace=True)
#     df.index.name = "Timestamp"


# # Define resampling function
# def resample_mixed(df):
#     numeric = df.select_dtypes(include="number")
#     categorical = df.select_dtypes(exclude="number")
#     numeric_resampled = numeric.resample("5min").mean() if not numeric.empty else pd.DataFrame(index=df.resample("5min").mean().index)
#     categorical_resampled = categorical.resample("5min").first() if not categorical.empty else pd.DataFrame(index=df.resample("5min").mean().index)

#     return pd.concat([numeric_resampled, categorical_resampled], axis=1)

# basal_resampled = resample_mixed(basal)
# bolus_resampled = resample_mixed(bolus)
# cgm_resampled = resample_mixed(cgm)
# alarms_resampled = resample_mixed(alarms)

# merged = cgm_resampled \
#     .join(basal_resampled, rsuffix="_basal") \
#     .join(bolus_resampled, rsuffix="_bolus") \
#     .join(alarms_resampled, rsuffix="_alarms")

# # Fix columns with comma decimals
# columns_with_comma = [
#     'Hoeveelheid', "Invoer bloedglucose (mmol/l)", "Eerste toediening (eenh.)",
#     "Uitgebreide toediening (eenh.)", "Serienummer_bolus", "Invoer koolhydraatverbruik (g)",
#     "Koolhydraatratio", "Toegediende insuline (eenh.)_bolus", "Serienummer_alarms"
# ]

# for col in columns_with_comma:
#     try:
#         merged[col] = merged[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '').astype(float)
#     except Exception as e:
#         print(f"Fout bij conversie van kolom {col}: {e}")

# # Attempt full numeric conversion just to be safe
# for col in columns_with_comma:
#     try:
#         merged[col] = pd.to_numeric(merged[col], errors='coerce')
#     except:
#         pass

# redundant_columns = ["Serienummer",	"Percentage (%)", "Toegediende insuline (eenh.)", "Serienummer_basal", 'Eerste toediening (eenh.)',	'Uitgebreide toediening (eenh.)', 'Serienummer_bolus', 	'Serienummer_alarms']
# merged = merged.drop(columns=redundant_columns)


# merged.rename(columns={
#     'Hoeveelheid': 'Insuline units (basal)',
#     'Invoer bloedglucose (mmol/l)': 'BG_input (mmol/l)',
#     'Invoer koolhydraatverbruik (g)': 'Carbohydrates (g)',
#     'Koolhydraatratio': 'Carb ratio',
#     'Toegediende insuline (eenh.)_bolus': 'Insuline units (bolus)',
#     'Duur (minuten)': 'Duration (minutes)',
#     "CGM-glucosewaarde (mmol/l)": "Glucose value (mmol/l)",
#     "Alarm/Gebeurtenis": "Alarm"
# }, inplace=True)

# # Translate categorical values
# merged["Insulinetype"] = merged["Insulinetype"].replace({
#     "Gepland": "Scheduled",
#     "Tijdelijk": "Temporary",
#     "Onderbreken": "Suspended"
# })

# merged["Insulinetype_bolus"] = merged["Insulinetype_bolus"].replace({
#     "Normaal": "Normal"
# })


# merged.to_csv("Glucose_export.csv")


import pandas as pd

# Shared column settings
columns_with_comma = [
    'Hoeveelheid', "Invoer bloedglucose (mmol/l)", "Eerste toediening (eenh.)",
    "Uitgebreide toediening (eenh.)", "Serienummer_bolus", "Invoer koolhydraatverbruik (g)",
    "Koolhydraatratio", "Toegediende insuline (eenh.)_bolus", "Serienummer_alarms"
]

redundant_columns = ["Serienummer",	"Percentage (%)", "Toegediende insuline (eenh.)", "Serienummer_basal",
                     'Eerste toediening (eenh.)', 'Uitgebreide toediening (eenh.)',
                     'Serienummer_bolus', 'Serienummer_alarms']

rename_dict = {
    'Hoeveelheid': 'Insuline units (basal)',
    'Invoer bloedglucose (mmol/l)': 'BG_input (mmol/l)',
    'Invoer koolhydraatverbruik (g)': 'Carbohydrates (g)',
    'Koolhydraatratio': 'Carb ratio',
    'Toegediende insuline (eenh.)_bolus': 'Insuline units (bolus)',
    'Duur (minuten)': 'Duration (minutes)',
    "CGM-glucosewaarde (mmol/l)": "Glucose value (mmol/l)",
    "Alarm/Gebeurtenis": "Alarm"
}

# Shared resampling function
def resample_mixed(df):
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    index = df.resample("5min").mean().index
    numeric_resampled = numeric.resample("5min").mean() if not numeric.empty else pd.DataFrame(index=index)
    categorical_resampled = categorical.resample("5min").first() if not categorical.empty else pd.DataFrame(index=index)
    return pd.concat([numeric_resampled, categorical_resampled], axis=1)

# Full pipeline
def process_glucose_data(basal_path, bolus_path, alarms_path, cgm1_path, output_file, cgm2_path=None):
    basal = pd.read_csv(basal_path, skiprows=1)
    bolus = pd.read_csv(bolus_path, skiprows=1)
    alarms = pd.read_csv(alarms_path, skiprows=1)
    cgm1 = pd.read_csv(cgm1_path, skiprows=1)

    if cgm2_path:  # If a second CGM file is given
        cgm2 = pd.read_csv(cgm2_path, skiprows=1)
        cgm = pd.concat([cgm1, cgm2])
    else:
        cgm = cgm1


    for df in [basal, bolus, cgm, alarms]:
        df["Tijdstempel"] = pd.to_datetime(df["Tijdstempel"], dayfirst=True, errors="coerce")
        df.set_index("Tijdstempel", inplace=True)
        df.sort_index(inplace=True)
        df.index.name = "Timestamp"

    merged = resample_mixed(cgm) \
        .join(resample_mixed(basal), rsuffix="_basal") \
        .join(resample_mixed(bolus), rsuffix="_bolus") \
        .join(resample_mixed(alarms), rsuffix="_alarms")

    for col in columns_with_comma:
        try:
            merged[col] = merged[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '').astype(float)
        except Exception as e:
            print(f"Fout bij conversie van kolom {col}: {e}")
        try:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
        except:
            pass

    merged = merged.drop(columns=[col for col in redundant_columns if col in merged.columns], errors='ignore')
    merged.rename(columns=rename_dict, inplace=True)

    if "Insulinetype" in merged.columns:
        merged["Insulinetype"] = merged["Insulinetype"].replace({
            "Gepland": "Scheduled",
            "Tijdelijk": "Temporary",
            "Onderbreken": "Suspended"
        })

    if "Insulinetype_bolus" in merged.columns:
        merged["Insulinetype_bolus"] = merged["Insulinetype_bolus"].replace({
            "Normaal": "Normal"
        })

    merged.to_csv(output_file)
    print(f"Saved to {output_file}")


# Training set
process_glucose_data(
    "data/Insulin data/basal_data_1.csv",
    "data/Insulin data/bolus_data_1.csv",
    "data/alarms_data_1.csv",
    "data/cgm_data_1.csv",
    "Glucose_export.csv",
    cgm2_path="data/cgm_data_2.csv"
)

# Test set
process_glucose_data(
    "test_data/Insulin data/basal_data_1.csv",
    "test_data/Insulin data/bolus_data_1.csv",
    "test_data/alarms_data_1.csv",
    "test_data/cgm_data_1.csv",
    "Glucose_export_test.csv"
)
