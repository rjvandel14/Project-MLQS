import pandas as pd
import numpy as np
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from outlierschapter.KalmanFilters import KalmanFilters
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def find_best_glucose_window(df, imputer, windows=[5, 10, 15, 20, 30, 45, 50, 55, 60]):
    col = 'Glucose value (mmol/l)'
    original = df.copy()

    known = df[df[col].notna()]
    test_indices = known.sample(frac=0.1, random_state=42).index
    true_values = df.loc[test_indices, col]

    errors = {}

    for w in windows:
        test_df = df.copy()
        test_df.loc[test_indices, col] = np.nan

        imputed = imputer.impute_glucose_sliding(test_df, window_minutes=w)
        preds = imputed.loc[test_indices, col]

        error = mean_absolute_error(true_values, preds)
        errors[w] = error
        print(f"Window {w} min → MAE: {error:.3f}")

    best_window = min(errors, key=errors.get)
    print(f"\n Best window: {best_window} min (MAE = {errors[best_window]:.3f})")
    return best_window, errors

class DomainSpecificImputation:

    def __init__(self):
        self.kf = KalmanFilters()
    
    def impute_glucose_sliding_dynamic(self, df, max_window_minutes=60, step=5, min_points=2):
        col = 'Glucose value (mmol/l)'
        if col not in df.columns:
            return df

        df = df.copy()
        df['time_index'] = (df.index - df.index[0]).total_seconds() / 60  # minutes since start
        missing_indices = df[df[col].isna()].index

        for t in missing_indices:
            center = df.at[t, 'time_index']

            # Try increasingly larger windows
            for window in range(step, max_window_minutes + step, step):
                lower_bound = center - window
                upper_bound = center + window

                window_df = df[(df['time_index'] >= lower_bound) & (df['time_index'] <= upper_bound) & (df[col].notna())]

                if len(window_df) >= min_points:
                    X = window_df[['time_index']]
                    y = window_df[col]
                    model = LinearRegression()
                    model.fit(X, y)
                    df.at[t, col] = model.predict([[center]])[0]
                    break  # Stop expanding the window after successful imputation

            # Optional fallback: interpolate if no good window found
            if pd.isna(df.at[t, col]):
                df.at[t, col] = df[col].interpolate(method="linear").at[t]

        df.drop(columns='time_index', inplace=True)
        return df
    def impute_basal_insulin(self, df):
        # Work directly on the real columns
        last_type = None
        last_units = None
        last_time = None
        last_duration = 0
        last_end_time = None

        for t, row in df[['Insuline units (basal)', 'Duration (minutes)', 'Insulinetype']].iterrows():
            val = row['Insulinetype']
            is_missing = pd.isna(val) or str(val).strip().lower() in ['', 'nan']

            # Case 1: new segment starts
            if not is_missing and pd.notna(row['Duration (minutes)']):
                last_type = row['Insulinetype']
                last_units = row['Insuline units (basal)']
                last_time = t
                last_duration = row['Duration (minutes)']
                last_end_time = t + pd.Timedelta(minutes=last_duration)

                # Fill current row
                df.at[t, 'Insulinetype'] = last_type
                df.at[t, 'Insuline units (basal)'] = last_units

                # Fill "Busy" rows
                for i in range(1, math.ceil(last_duration / 5)):
                    busy_t = t + pd.Timedelta(minutes=5 * i)
                    if busy_t in df.index:
                        busy_val = df.at[busy_t, 'Insulinetype']
                        if pd.isna(busy_val) or str(busy_val).strip().lower() in ['', 'nan']:
                            df.at[busy_t, 'Insulinetype'] = f"Busy {last_type}"
                            df.at[busy_t, 'Insuline units (basal)'] = last_units

            # Case 2: after segment ends, no new segment → fill with "Scheduled", -1
            elif is_missing and last_end_time is not None and t >= last_end_time:
                df.at[t, 'Insulinetype'] = "Scheduled"
                df.at[t, 'Insuline units (basal)'] = np.nan

        # Case 3: Suspended always has insulin = 0
        suspended_mask = df['Insulinetype'].str.contains("Suspended", na=False)
        df.loc[suspended_mask, 'Insuline units (basal)'] = 0.0

        # Case 4: Apply Kalman filter separately per type
        for typ in ['Scheduled', 'Temporary']:
            mask = df['Insulinetype'] == typ
            if mask.any():
                df.loc[mask] = self.kf.apply_kalman_filter(df.loc[mask], 'Insuline units (basal)')
        
        # Propagate insulin values from Scheduled/Temporary to Busy Scheduled/Temporary
        for typ in ['Scheduled', 'Temporary']:
            last_value = None
            for t in df.index:
                type_val = df.at[t, 'Insulinetype']
                if type_val == typ:
                    last_value = df.at[t, 'Insuline units (basal)']
                elif type_val == f"Busy {typ}" and last_value is not None:
                    df.at[t, 'Insuline units (basal)'] = last_value

        return df
    
    def impute_bolus_insulin(self, df):
        # Define bolus-related columns
        bolus_fields = ['BG_input (mmol/l)', 'Insulinetype_bolus', 'Carbohydrates (g)', 'Carb ratio', 'Insuline units (bolus)']

        # Compute median insulin for correction boluses (carbs = 0)
        correction_doses = df[(df['Carbohydrates (g)'] == 0) & df['Insuline units (bolus)'].notna()]
        median_insulin = correction_doses['Insuline units (bolus)'].median()

        for t, row in df[bolus_fields].iterrows():
            all_missing = all(pd.isna(row[field]) or str(row[field]).strip().lower() in ['', 'nan'] for field in bolus_fields)

            # Case 1: No bolus at all → fill with zeros and "No Bolus"
            if all_missing:
                df.at[t, 'BG_input (mmol/l)'] = 0
                df.at[t, 'Insulinetype_bolus'] = "No Bolus"
                df.at[t, 'Carbohydrates (g)'] = 0
                df.at[t, 'Carb ratio'] = 0
                df.at[t, 'Insuline units (bolus)'] = 0

            else:
                # Case 2: Some bolus present → fill selectively
                df.at[t, 'Insulinetype_bolus'] = "Normal"  # Always assume normal if any bolus info is present

                # BG imputation from CGM value
                if pd.isna(row['BG_input (mmol/l)']) and 'Glucose value (mmol/l)' in df.columns:
                    df.at[t, 'BG_input (mmol/l)'] = df.at[t, 'Glucose value (mmol/l)']

                carbs = row['Carbohydrates (g)']
                ratio = row['Carb ratio']
                insulin = row['Insuline units (bolus)']

                # Carbs = 0, insulin present → correction bolus
                if carbs == 0 and pd.notna(insulin):
                    df.at[t, 'Carb ratio'] = 0

                # Carbs = 0, insulin missing → impute insulin with median correction dose
                if carbs == 0 and pd.isna(insulin):
                    df.at[t, 'Insuline units (bolus)'] = median_insulin
                    df.at[t, 'Carb ratio'] = 0

                # Impute missing Carb Ratio
                elif pd.isna(ratio) and pd.notna(carbs) and pd.notna(insulin) and insulin != 0:
                    inferred_ratio = carbs / insulin
                    df.at[t, 'Carb ratio'] = 12 if abs(inferred_ratio - 12) < abs(inferred_ratio - 14) else 14

                # Impute missing Insulin from carbs and ratio
                elif pd.notna(carbs) and pd.notna(ratio) and pd.isna(insulin):
                    df.at[t, 'Insuline units (bolus)'] = carbs / ratio

                # Impute missing Carbs from insulin and ratio
                elif pd.notna(insulin) and pd.notna(ratio) and pd.isna(carbs):
                    df.at[t, 'Carbohydrates (g)'] = insulin * ratio

                # Fallback: if ratio is missing, assign most common
                if pd.isna(df.at[t, 'Carb ratio']):
                    df.at[t, 'Carb ratio'] = 12  # or 14, depending on dominant value in your dataset

        return df

    def impute_alarms(self, df):
        if 'Alarm' in df.columns:
            df['Alarm'] = df['Alarm'].apply(
                lambda x: "No alarm" if pd.isna(x) or str(x).strip().lower() in ['', 'nan'] else x
            )
        return df
