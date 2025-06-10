##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 3                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from outlierschapter.OutlierDetection import DistributionBasedOutlierDetection
from outlierschapter.OutlierDetection import DistanceBasedOutlierDetection
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.
RESULT_FNAME = 'result_outliers.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()

    # Load dataset
    dataset = pd.read_csv("Glucose_export.csv", parse_dates=["Tijdstempel"])
    dataset.set_index("Tijdstempel", inplace=True)

    # We'll create an instance of our visualization class to plot the results.
    DataViz = VisualizeDataset(__file__)
    

    # Step 1: Let us see whether we have some outliers we would prefer to remove.

    # Determine the columns we want to experiment on.
    outlier_columns = ['Glucose value (mmol/l)', 'Insuline units (basal)', "Duration (minutes)", "BG_input (mmol/l)",  
                       "Carbohydrates (g)", "Carb ratio", "Insuline units (bolus)"
                       ]
    
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

            # Determine threshold at 1% lowest likelihood (you can adjust this)
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

        # We use Chauvenet's criterion for the final version and apply it to all but the label data...
        for col in [c for c in dataset.columns if not 'label' in c]:

            print(f'Measurement is now: {col}')
            dataset = OutlierDistr.chauvenet(dataset, col, FLAGS.C)
            dataset.loc[dataset[f'{col}_outlier'] == True, col] = np.nan
            del dataset[col + '_outlier']

        dataset.to_csv(RESULT_FNAME)
print(f"Outlier-cleaned data saved to: {RESULT_FNAME}")


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()


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
