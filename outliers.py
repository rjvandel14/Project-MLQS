# Used and edited:
# Chapter 3 MLQS Springer by Mark Hoogendoorn and Burkhardt Funk (2017)   


from util.VisualizeDataset import VisualizeDataset
from outlierschapter.OutlierDetection import DistributionBasedOutlierDetection
from outlierschapter.OutlierDetection import DistanceBasedOutlierDetection
import pandas as pd
import numpy as np
import argparse

# Set up file names and locations.
RESULT_FNAME = 'result_outliers.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

# final model selections per attribute

final_methods = {
    "Duration (minutes)": {
        "method": "chauvenet",
        "params": {"C": 5}
    },
    "Insuline units (basal)": {
        "method": "distance",
        "params": {"dmin": 0.1, "fmin": 0.99}
    },
    "Carbohydrates (g)": {
        "method": "mixture",
        "params": {"n_est": 2, "quantile": 0.01}
    },
    "BG_input (mmol/l)": {
        "method": "mixture",
        "params": {"n_est": 2, "quantile": 0.05}
    },
    "Insuline units (bolus)": {
        "method": "mixture",
        "params": {"n_est": 3, "quantile": 0.01}
    },
    "Glucose value (mmol/l)": {
        "method": "chauvenet",
        "params": {"C": 1.5}
    }
}


def main():
    print_flags()

    # Load dataset
    dataset = pd.read_csv(FLAGS.input, parse_dates=["Timestamp"])
    dataset.set_index("Timestamp", inplace=True)


    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)

    # Step 1: Let us see whether we have some outliers we would prefer to remove.
    
    outlier_columns = list(final_methods.keys())
 

    # Create the outlier classes.
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()
    #chose one of the outlier methods: chauvenet, mixture, distance or LOF via the argument parser at the bottom of this page. 


    if FLAGS.mode == 'chauvenet':

        # And investigate the approaches for all relevant attributes.
        for col in outlier_columns:

            print(f"Applying Chauvenet outlier criteria for column {col}")

            # And try out all different approaches. Note that we have done some optimization
            # of the parameter values for each of the approaches by visual inspection.
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            print(f"{col} → {dataset[col + '_outlier'].sum()} outliers detected.")

            DataViz.plot_binary_outliers(
                dataset, col, col + '_outlier')

         
    elif FLAGS.mode == 'mixture':
        for col in outlier_columns:
            print(f"Applying mixture model for column {col}")
            dataset = OutlierDistr.mixture_model(dataset, col)

            # Determine threshold at 5% lowest likelihood 
            threshold = dataset[col + '_mixture'].quantile(0.01)

            # Flag outliers: those with likelihood below the threshold
            dataset[col + '_outlier'] = dataset[col + '_mixture'] < threshold

            # Count and print number of outliers
            num_outliers = dataset[col + '_outlier'].sum()
            print(f"{col} → {num_outliers} outliers detected (bottom 1% of likelihoods)")

            # Plot just like in Chauvenet
            DataViz.plot_binary_outliers(dataset, col, col + '_outlier')
            

    elif FLAGS.mode == 'distance':
        for col in outlier_columns:
            try:
                dataset = OutlierDist.simple_distance_based(
                    dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
                print(f"{col} → {dataset['simple_dist_outlier'].sum()} outliers detected.")
                DataViz.plot_binary_outliers(
                    dataset, col, 'simple_dist_outlier')
            except MemoryError as e:
                print(
                    'Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')


    elif FLAGS.mode == 'LOF':
        for col in outlier_columns:
            try:
                dataset_small = dataset.sample(5000)

                # Compute LOF scores
                dataset_small = OutlierDist.local_outlier_factor(
                    dataset_small, [col], 'euclidean', FLAGS.K)

                # Define a threshold, e.g. LOF > 1.5 is considered an outlier
                lof_threshold = 1.5
                dataset_small[col + '_outlier'] = dataset_small['lof'] > lof_threshold

                # Print number of outliers
                print(f"{col} → {dataset_small[col + '_outlier'].sum()} outliers detected using LOF (LOF > {lof_threshold}).")

                # Plot binary outliers
                DataViz.plot_binary_outliers(dataset_small, col, col + '_outlier')

            except MemoryError as e:
                print('Not enough memory available for LOF...')
                print('Skipping.')

    
    
    elif FLAGS.mode == 'final':
        for col, config in final_methods.items():
            method = config["method"]
            print(f"Processing column '{col}' using method: {method}")

            if method == "chauvenet":
                C = config["params"]["C"]
                dataset = OutlierDistr.chauvenet(dataset, col, C)
                dataset.loc[dataset[f"{col}_outlier"], col] = np.nan
                del dataset[f"{col}_outlier"]

        
            elif method == "mixture":
                n_est = config["params"]["n_est"]
                quantile = config["params"]["quantile"]

                # Fit mixture model and get likelihoods
                dataset = OutlierDistr.mixture_model(dataset, col, n_components=n_est)
                
                # Fix: threshold on the *likelihoods*, not on class labels
                threshold = dataset[col + "_mixture"].quantile(quantile)
                dataset[col + "_outlier"] = dataset[col + "_mixture"] < threshold

                dataset.loc[dataset[col + "_outlier"], col] = np.nan
                dataset.drop(columns=[col + "_mixture", col + "_outlier"], inplace=True)


            elif method == "distance":
                dmin = config["params"]["dmin"]
                fmin = config["params"]["fmin"]
                dataset = OutlierDist.simple_distance_based(dataset, [col], "euclidean", dmin, fmin)
                dataset.loc[dataset["simple_dist_outlier"].fillna(False), col] = np.nan
                dataset.drop(columns=["simple_dist_outlier"], inplace=True)

    dataset.to_csv(FLAGS.output)
    print(f"Outlier-cleaned data saved to: {FLAGS.output}")



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='new data/Glucose_export.csv',
                    help="Input CSV file with glucose data")
    
    parser.add_argument('--output', type=str, default='new data/result_outliers.csv',
                    help="Output CSV file name")

    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter", choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])

    parser.add_argument('--C', type=float, default=2,
                        help="Chauvenet: C parameter")
   
    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor:  K is the number of neighboring points considered")

    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based:  dmin is ... ")

    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based:  fmin is ... ")

    FLAGS, unparsed = parser.parse_known_args()

    main()

