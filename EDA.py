
from util.VisualizeDataset import VisualizeDataset
from util import util
import pandas as pd


df = pd.read_csv("Glucose_export.csv", parse_dates=["Tijdstempel"], decimal=",")

# Plot the data
DataViz = VisualizeDataset(__file__)

# Boxplot
DataViz.plot_dataset_boxplot(df, ['CGM-glucosewaarde (mmol/l)', 'Serienummer','Duur (minuten)','Percentage (%)', 'Toegediende insuline (eenh.)', 'Serienummer_basal', 'Insulinetype','Hoeveelheid','Invoer bloedglucose (mmol/l)','Eerste toediening (eenh.)','Uitgebreide toediening (eenh.)','Serienummer_bolus','Insulinetype_bolus','Invoer koolhydraatverbruik (g)','Koolhydraatratio',"Toegediende insuline (eenh.)_bolus","Serienummer_alarms","Alarm/Gebeurtenis"])

# Plot all data
DataViz.plot_dataset(df, ['CGM-glucosewaarde (mmol/l)', 'Serienummer','Duur (minuten)','Percentage (%)', 'Toegediende insuline (eenh.)', 'Serienummer_basal', 'Insulinetype','Hoeveelheid','Invoer bloedglucose (mmol/l)','Eerste toediening (eenh.)','Uitgebreide toediening (eenh.)','Serienummer_bolus','Insulinetype_bolus','Invoer koolhydraatverbruik (g)','Koolhydraatratio',"Toegediende insuline (eenh.)_bolus","Serienummer_alarms","Alarm/Gebeurtenis"],
                                ['like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like', 'like', 'like', 'like', 'like', 'like', 'like','like', 'like','like'],
                                ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'])

# And print a summary of the dataset.
util.print_statistics(df)