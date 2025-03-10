# Importing required libraries 

import os
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, weibull_min, invweibull
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

import hdbscan

from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import galactocentric_frame_defaults, SkyCoord, Galactic, CartesianRepresentation
from astropy.table import Table, Column

# Set the galactocentric frame defaults
galactocentric_frame_defaults.set('latest')

###=============================================================== Function to visualize two dimensions at a time==========================
def plot_2d(dim1, dim2, ax, data, cluster_labels, MCflag = 0, MC_thr = 500, prob_thr =0.5,prob = None, selected_labels = None,leg_flag = 1,alpha1 = 1, alpha2 = 1):
    """
    Create a 2D scatter plot in a specified axis, highlighting cluster members.

    Parameters:
    dim1 (str): The label of the x-axis dimension.
    dim2 (str): The label of the y-axis dimension.
    ax (matplotlib.axes.Axes): The axis to draw the plot on.
    data (pd.DataFrame): The data containing the features.
    cluster_labels (pd.Series): The cluster labels assigned to each data point or may contain membership record if MCflag =1
    MCflag (0 or 1): Decides whether Monte-Carlo sampling has been applied.
    """
    #  "#18dbcb" - light blue type color
    # '#f00ea1' - light pink
    #  #D3D3D3 - light grey
    # #C0C0C0 - silver
    gcolor = ["orange","#18dbcb", "#93f707", "#93f707","#18dbcb", "white", "#f00ea1", "goldenrod", "blue", "sienna", "#357362"] * 20
    gmarker = ["o", "D", "^", "v", ">", "<", "p", "s", "d", 'o'] * 20

    ax.scatter(data.loc[:, dim1], data.loc[:, dim2], c='#C0C0C0', alpha=0.5 , s = 10,label = 'Outliers')
    cluster_data = []

    if MCflag == 1:
        ind = data['source_id'].isin(cluster_labels['source_id'])
        cluster_data = data[ind]
        print(f"Total detected members are: {len(cluster_data)}")
        # print(f"True members: {np.sum(cluster_data.iloc[:,13])}")
        # print(f"Total False positive detections: {len(cluster_data) - np.sum(cluster_data.iloc[:,13])}")
        t_IR = data[ind][~(data[ind]['ID'].astype(str)== '        ')]
        t_IR_total = data[~(data['ID'].astype(str)== '        ')]
        print(f"The no. of SFOG YSOs: {len(t_IR)} out of {len(t_IR_total)}")

        t_optical = data[ind][(data[ind]['ID'].astype(str)== '        ')]
        t_optical_total = data[(data['ID'].astype(str)== '        ')]
        print(f"The no. of Gaia YSOs: {len(t_optical)} out of {len(t_optical_total)}")

        ax.set_facecolor('black')
        # ax.scatter(cluster_data.loc[:, dim1], cluster_data.loc[:, dim2], alpha1=alpha1,s=3, c=gcolor[0], marker=gmarker[0])
        # ax.scatter(cluster_data.loc[:, dim1], cluster_data.loc[:, dim2], alpha=alpha1,s=15, c=gcolor[1], marker=gmarker[1],label = 'member YSOs')

        ax.scatter(t_optical.loc[:, dim1], t_optical.loc[:, dim2], alpha=alpha1,s=15, c=gcolor[1], marker=gmarker[1],label = 'Gaia YSOs')
        ax.scatter(t_IR.loc[:, dim1], t_IR.loc[:, dim2], alpha=alpha2,s=13, c=gcolor[0], marker=gmarker[0], label = 'Gaia matched SFOG YSOs')
        
        
        
    if MCflag == 0:
        for cluster_label in set(cluster_labels):
            if cluster_label != -1:  # Ignore outliers and check probability
                cluster_data = data[(cluster_labels == cluster_label) & (prob > prob_thr)]
                print(len(cluster_data))
                ax.set_facecolor('black')
                ax.scatter(cluster_data.loc[:, dim1], cluster_data.loc[:, dim2], label=f'Cluster {cluster_label+1}', alpha=alpha1,s=3, c=gcolor[cluster_label], marker=gmarker[cluster_label])    

    if MCflag == 2:
        for label in selected_labels:
            cluster_data = data[(cluster_labels == label) & (prob > prob_thr)]
            print(len(cluster_data))
            ax.set_facecolor('black')
            ax.scatter(cluster_data.loc[:, dim1], cluster_data.loc[:, dim2], label=f'Cluster {label+1}', alpha=alpha1,s=3, c=gcolor[label], marker=gmarker[label])
    # # Highlight cluster members
    # for cluster_label,p in enumerate(cluster_labels,prob):
    #     if cluster_label != -1:  # Ignore outliers
    #         cluster_data = data[(cluster_labels == cluster_label) and p > 0.8]
    #         ax.scatter(cluster_data.loc[:, dim1], cluster_data.loc[:, dim2], label=f'Cluster {cluster_label}', alpha=1, s=3, c=gcolor[cluster_label], marker=gmarker[cluster_label])

    ax.set_xlabel(f'{dim1}', fontsize=13)
    ax.set_ylabel(f'{dim2}', fontsize=13)
    ax.tick_params(axis='both', labelsize=15)

    if leg_flag:
        ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.35),fontsize = 17)
    ax.set_title(f'{dim1}-{dim2} space', fontsize= 20)

    return cluster_data

###===============================================================Normalization Function======================================

def normalization_2sigma(inputs, inputs0,quantile_range_dist, quantile_range_pm):

    inputs_norm = np.empty_like(inputs)
    
    for i in range(inputs.shape[1]):
        if (i == 3) or (i == 4):
            norm_range = 2*np.abs(np.percentile(inputs0[:,i],quantile_range_pm[1])-np.percentile(inputs0[:,i],quantile_range_pm[0]))
            inputs_norm[:,i] = (inputs[:,i])/norm_range
            inputs_norm[:,i] = inputs_norm[:,i] - np.median(inputs_norm[:,i])

        else:    
            norm_range = np.percentile(inputs0[:,i],quantile_range_dist[1])-np.percentile(inputs0[:,i],quantile_range_dist[0])
              
            inputs_norm[:,i] = (inputs[:,i])/norm_range
            inputs_norm[:,i] = inputs_norm[:,i] - np.median(inputs_norm[:,i])
   
    return inputs_norm

# Startegy 1
# def normalization_2sigma(inputs, inputs0,quantile_range):

#     inputs_norm = np.empty_like(inputs)
    
#     for i in range(inputs.shape[1]):
#         if (i == 3) or (i == 4):
#             norm_range = np.abs(np.percentile(inputs[:,i],quantile_range[1])-np.percentile(inputs[:,i],quantile_range[0]))
                
#             inputs_norm[:,i] = (inputs[:,i])/norm_range
#             inputs_norm[:,i] = inputs_norm[:,i] - np.median(inputs_norm[:,i])

#         else:    
#             norm_range = np.percentile(inputs0[:,i],quantile_range[1])-np.percentile(inputs0[:,i],quantile_range[0])
                
#             inputs_norm[:,i] = (inputs[:,i])/norm_range
#             inputs_norm[:,i] = inputs_norm[:,i] - np.median(inputs_norm[:,i])
   
#     return inputs_norm

###===============================================================Function to get real projected space velocities======================================

def projected_velocity(dist, pmra, pmdec):
    dist = dist * u.pc
    pmra = pmra * (1e-3) * u.arcsec 
    pmdec = pmdec * (1e-3) * u.arcsec 

    vel_ra = ((dist * np.tan(pmra)).to(u.km)/u.year).to(u.km/u.s).value
    vel_dec = ((dist * np.tan(pmdec)).to(u.km)/u.year).to(u.km/u.s).value

    return vel_ra, vel_dec

def space_coords(ra, dec, final_dist):
    # Convert spherical coordinates to Cartesian coordinates
    c = coord.SkyCoord(ra=ra * u.degree,
                       dec=dec * u.degree,
                       distance=final_dist * u.pc,
                       frame='icrs')

    # Transform to Galactocentric coordinates
    galactocentric_cartesian = c.transform_to(coord.Galactocentric(galcen_distance=8.178*u.kpc)) 

    # Extract Cartesian coordinates
    X_column = Column(data=galactocentric_cartesian.x.value, name='X', dtype='float64')
    Y_column = Column(data=galactocentric_cartesian.y.value, name='Y', dtype='float64')
    Z_column = Column(data=galactocentric_cartesian.z.value, name='Z', dtype='float64')

    return np.array(X_column), np.array(Y_column), np.array(Z_column)
###===============================================================Function to create mock catalogs======================================
def catalog_creator(data):
    """
    Create a mock catalog based on input data.

    Parameters:
    data (pd.DataFrame): The input data containing percentiles and other information.

    Returns:
    pd.DataFrame: A DataFrame containing the generated mock catalog with scaled features.
    """

    # Extract percentile columns from the input data
    d_50 = data['final_dist'].values
    d_16 = data['final_dist_lo'].values
    d_84 = data['final_dist_hi'].values

    ra = data['ra'].values
    dec = data['dec'].values

    # Parameters for inverse Weibul distribution
    # Estimate shape parameter (k) using the 16th percentile
    k = np.log(np.log(0.84) / np.log(0.16)) / np.log(d_16 / d_84)

    # Estimate scale parameter (λ) using the 50th percentile
    λ = d_50 * np.log(2)**(1/k) 

    # # # Generate a unique seed based on the process ID
    # pid = os.getpid()
    # seed = pid

    # Set the random seed
    np.random.seed()
    
   # Generate random samples from the Weibull distribution
    dist_sample = invweibull.rvs(c=k, scale=λ, size=len(data))
    # Generate random samples from the Gaussian distribution

     # Set the random seed
    np.random.seed()
    pmra_sample = norm.rvs(loc = data['pmra'], scale = data['pmra_error'], size = len(data))

     # Set the random seed
    np.random.seed()
    pmdec_sample = norm.rvs(loc = data['pmdec'], scale = data['pmdec_error'], size = len(data))

    vel_ra,vel_dec = projected_velocity(dist_sample, pmra_sample, pmdec_sample)

    X,Y,Z = space_coords(ra, dec, dist_sample)
    

    mock_catalog = {
        'X': X,
        'Y': Y,
        'Z': Z,
        'pmra': vel_ra,
        'pmdec': vel_dec
    }


    # Create a DataFrame from the dictionary
    unscaled_features = pd.DataFrame(mock_catalog)

    inputs = unscaled_features.to_numpy()

    inputs_for_norm = np.empty_like(inputs)
    inputs_for_norm[:,0:5] = np.abs(inputs[:,0:5])
    # unscaled_features['mock_distances'] = np.log10(unscaled_features['mock_distances'])

    # ===========================Feature scaling====================================
    scaled_features = np.zeros_like(unscaled_features)  # Create an empty array

    # scaled_features[:,0:3] =  inputs[:,0:3]
    # scaled_features[:,3:5] =  inputs[:,3:5]
    
    scaled_features = normalization_2sigma(inputs,inputs_for_norm,(0,50),(0,99.5))

    # scaled_features[:,0:3] = 100*scaled_features[:,0:3]
    # scaler0 =  MinMaxScaler()
    # scaler1 = RobustScaler(quantile_range=(2.5, 97.5))
    # scaler2 = StandardScaler()

    # scaled_features[:,3:5] = scaler2.fit_transform(inputs[:,3:5])
    # scaled_features[:, 0:3] = scaler1.fit_transform(inputs[:, 0:3])  # Use .iloc for DataFrame
    # # scaled_features[:, 2] = scaled_features[:, 2]/2

    
    # scaled_features[:, 3:5] = scaler2.fit_transform(unscaled_features.iloc[:, 3:5])  # Use .iloc for DataFrame
    # dist_sample, pmra_sample, pmdec_sample
    return scaled_features 

###============================================== Displaying progress bar===============================================================
def print_loading_bar(progress, block_char="█"):
    bar_length = 100
    num_blocks = int(bar_length * progress)
    bar = block_char * num_blocks + " " * (bar_length - num_blocks)
    print(f"[{bar}] {progress*100:.1f}%", end="\r")
    
    
###=============================================Running HDBSCAN with Monte-Carlo error sampling========================================
def run_hdbscan(data, n = 1000, cluster_no = 0, min_cluster_size=10, prob_thr = 0.5):
    """o-0
    Run HDBSCAN clustering on the data for multiple iterations and
    accumulate membership records.

    Parameters:
    data (numpy.ndarray): The input data for clustering.
    n (int): The number of iterations to run HDBSCAN.
    cluster_no (int): The target cluster number for MCHDBSCAN.

    Returns:
    numpy.ndarray: The membership record indicating cluster membership.
    """
    membership_record = np.zeros(len(data))
    clusterer = hdbscan.HDBSCAN(min_cluster_size= min_cluster_size, cluster_selection_method='eom',allow_single_cluster = True)

    prob_thr = prob_thr

    for x in range(n):
        inp = catalog_creator(data)

        clusterer.fit(inp)
        cluster_labels = clusterer.labels_
        prob = clusterer.probabilities_
        ##############################################
        ## Plotting the members
        ## Part is for testing. Uncomment if required
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        # plot_2d('ra', 'dec', ax1, data, cluster_labels, prob = prob, prob_thr = prob_thr)
        # plot_2d('pmra', 'pmdec', ax2, data, cluster_labels, prob = prob, prob_thr = prob_thr)
        # plt.show()
        ##############################################

        outliers = (cluster_labels == -1)
        
        # Count the number of members in each cluster (excluding outliers)
        unique_clusters, cluster_counts = np.unique(cluster_labels[~outliers], return_counts=True)
        
        # Sort the cluster labels based on the counts in descending order
        sorted_cluster_labels = unique_clusters[np.argsort(cluster_counts)[::-1]]


	    # Create a mapping from old labels to new labels (0, 1, 2, ...)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_labels)}
        sorted_labels = np.full(cluster_labels.shape, -1)
        sorted_labels[~outliers] = np.array([label_mapping[label] for label in cluster_labels[~outliers]])

        
        
        # The line of code below is NumPy's `np.where` function to assign values to the `cluster_labels` 
        # array based on a condition. Here's what each part does:
        # (sorted_labels != cluster_no)` checks if the elements of `sorted_labels` are not equal to `cluster_no`.
        # (prob < prob_thr)` checks if the elements of `prob` are less than `prob_thr`.
        # ((sorted_labels != cluster_no) | (prob < prob_thr))` combines the two conditions using the 
        # logical OR (`|`) operator, resulting in an array of boolean values indicating whether either 
        # condition is true for each element.
        # np.where(condition, x, y)` returns an array where elements are taken from `x` where `condition` is true 
        # and from `y` where it is false. In this case, when the condition is true, 
        # it assigns `0`, and when false, it assigns `1`. 

        # So, the line sets `cluster_labels` to `0` where the condition is true and `1` where it is false, based on 
        # the two conditions combined with logical OR.
        cluster_labels = np.where(
            ((sorted_labels != cluster_no) | (prob < prob_thr)), 0, 1
        )

        ##############################################
        ## Plotting the members with reassigned labels
        ## Part is for testing. Uncomment if required
        # print(" This is plot are after label reassignment\n")

        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        # plot_2d('ra', 'dec', ax1, data, cluster_labels, prob = prob, prob_thr = prob_thr)
        # plot_2d('pmra', 'pmdec', ax2, data, cluster_labels, prob = prob, prob_thr = prob_thr)
        # plt.show()
        ##############################################
        membership_record += cluster_labels

        # Print loading bar
        if (x + 1) % 10 == 0:
            progress = (x + 1) / n
            print_loading_bar(progress, "█")

    # Print a new line after the loop
    print()
    
    return membership_record


def cluster_iteration(data, clusterer, cluster_no = 0, prob_thr = 0.25):
    """Run a single iteration of HDBSCAN clustering."""
    clusterer.fit(catalog_creator(data))  # inp = catalog_creator(data)

    cluster_labels = clusterer.labels_
    prob = clusterer.probabilities_


    outliers = (cluster_labels == -1)
        
    # Count the number of members in each cluster (excluding outliers)
    unique_clusters, cluster_counts = np.unique(cluster_labels[~outliers], return_counts=True)
        
    # Sort the cluster labels based on the counts in descending order
    sorted_cluster_labels = unique_clusters[np.argsort(cluster_counts)[::-1]]


	# Create a mapping from old labels to new labels (0, 1, 2, ...)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_labels)}
    sorted_labels = np.full(cluster_labels.shape, -1)
    sorted_labels[~outliers] = np.array([label_mapping[label] for label in cluster_labels[~outliers]])

        
        
    # Combine the conditions into a single array
    cluster_labels = np.where(
        ((sorted_labels != cluster_no) | (prob < prob_thr)), 0, 1
        )
    
    # # Print loading bar
    # if (x + 1) % 10 == 0:
    #     progress = (x + 1) / n
    #     print_loading_bar(progress, "█")
    
    return cluster_labels


 
def run_hdbscan_parallel(data, n=10000, cluster_no=0, min_cluster_size=10, prob_thr=0.25):
    """Run HDBSCAN clustering on the data for multiple iterations in parallel."""
    membership_record = np.zeros(len(data))
    clusterer = hdbscan.HDBSCAN(min_cluster_size= min_cluster_size, cluster_selection_method='eom',allow_single_cluster = True)

    # # # Define the number of iterations per process
    iterations_per_process = n
    # if iterations_per_process == 0:
    #     iterations_per_process = 1

    # iterations_per_process = n/10
    # iterations_per_process = 1000
    # Create a pool of worker processes
    nSlots = int(os.getenv('NSLOTS'))
    with multiprocessing.Pool(nSlots+20) as pool:
        # Map the cluster_iteration function to input data for parallel processing
        results = pool.starmap(cluster_iteration, [(data, clusterer, cluster_no, prob_thr)] * iterations_per_process)
        pool.close()
        
    # Aggregate the results from all processes
    for result in results:
        membership_record += result

    return membership_record