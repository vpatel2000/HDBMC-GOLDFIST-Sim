import os
import gc
import importlib
import numpy as np
import pandas as pd
import configparser
from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Local imports
import HDBSCAN_func
import cluster_model

# Reload the module if needed
importlib.reload(HDBSCAN_func)
from HDBSCAN_func import *

importlib.reload(cluster_model)
from cluster_model import *


primary_dir = config["paths"]["primary_dir"]
sim_dir = config["paths"]["sim_dir"]

def monte_carlo_spectra(cluster_name,runs, sim_cluster_final):
    """
    Perform Monte Carlo simulations to generate membership records, Monte Carlo spectrum and residual.

    Parameters:
    cluster_name (str): The name of the cluster.
    runs (int): The number of Monte Carlo runs.
    sim_cluster_final (simulate_cluster): An instance of the simulate_cluster class.

    Returns:
    tuple: A tuple containing the residual, T, f, t, and membership_record_100k.
    f: numpy array of number of false-postive YSO detections for a given Monte Carlo threshold.
    t: numpy array of number of true YSO detections for a given Monte Carlo threshold.
    T: numpy array of total number of YSO detections for a given Monte Carlo threshold. [i.e. T=f+t]
    membership_record_100k: dictionary containing the membership records for 100k iterations.
    """

    os.chdir(sim_dir)
    threshold_array = np.arange(0, 100500, 500)

    king = sim_cluster_final.king2d(Wpermit = 1)
    region_simulation = sim_cluster_final.add_noise(king, Wpermit=1)
    final_sim = sim_cluster_final.add_real_error(region_simulation, Wpermit = 1, update_permit = 1)
    
    T_obs = np.load(os.path.join(primary_dir, f"{cluster_name}/total_mem.npy"))
    cluster_no = cluster_name.split('_')[1]

    data_path = os.path.join(sim_dir,"Data",f"{cluster_name}_sim")
    table=Table.read(data_path+f"/required_sim_data_{cluster_name}.fits")
    
    sim_cluster_dir = os.path.join(sim_dir,f"{cluster_name}_sim")
    os.makedirs(sim_cluster_dir, exist_ok=True)
    os.chdir(sim_cluster_dir)

    data = table.to_pandas()

    #==================================================Creating Membership Records===================================

    print("Generating membership records...........")
    print(f"{os.cpu_count()} cores have been put to work! So relax...")  

    if  os.path.isfile(f"membership_record_100k_{cluster_name}.npy"):
    # File exists, load the dictionary
        membership_record_100k = np.load(f"membership_record_100k_{cluster_name}.npy", allow_pickle=True).item()
        print("100k dictionary loaded successfully.")
    else:
        print("100k dictionary does not exist. Saving new dictionary.")
        membership_record_100k = {}
        for i in range(runs):
            membership_record_100k[i] = run_hdbscan_parallel(data, n=100000, cluster_no = 0, prob_thr = 0.1)
            print(f'Run: {i+1}. Completed 100000 iterations!')
        # np.save(f'membership_record_100k_{cluster_name}.npy', membership_record_100k)

    false_pos = []
    true_mem = []

    for threshold in threshold_array:

        storage_100k = {}
        for i in range(len(membership_record_100k)):
            MC_thr = threshold
            storage_100k[f'iter_{i}'] = data[membership_record_100k[i] > MC_thr]

        # # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
        # # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

        # Get the keys from storage corresponding to iter_1 to iter_10
        keys = ['iter_' + str(i) for i in range(0, runs)]

        # Initialize common_members with the 'source_id' column from the first iteration
        common_members_100k = storage_100k[keys[0]][['source_id']]

        # Iterate over the keys and merge each 'source_id' column with common_members
        for key in keys[1:]:
            common_members_100k = pd.merge(common_members_100k, storage_100k[key][['source_id']], on='source_id', how='inner')

        np.save('true_mem.npy',np.array(true_mem))
        np.save('false_pos.npy',np.array(false_pos))

        ind = data['source_id'].isin(common_members_100k['source_id'])
        cluster_data = data[ind]

        true_mem.append(np.sum(cluster_data.iloc[:,13]))
        false_pos.append(len(cluster_data) - np.sum(cluster_data.iloc[:,13]))
            
    np.save('true_mem.npy',np.array(true_mem))
    np.save('false_pos.npy',np.array(false_pos))

    t = np.array(true_mem)
    f = np.array(false_pos)
    T = t + f
     
    residual = np.mean((T_obs - T)**2)

    print(residual)
    print(sim_cluster_final.center_distance, sim_cluster_final.member_fraction, sim_cluster_final.noise_lower, sim_cluster_final.noise_upper, sim_cluster_final.c, sim_cluster_final.pm_scale)
    print("\n")

    return residual, np.array(T) , np.array(f), np.array(t), membership_record_100k


def weighted_std_unbiased(input,w):
    """
    Calculate the unbiased weighted standard deviation.

    Parameters:
    input (numpy.ndarray): The input array.
    w (numpy.ndarray): The weights.

    Returns:
    numpy.ndarray: The unbiased weighted standard deviation.
    """
    if len(w[w!=0])>1:
        V1 = np.sum(w)
        V2 = np.sum(w**2)

        mu_star = np.average(input,axis=0, weights=w)  # Weighted mean
        num = np.sum(w.reshape(-1,1)*((input-mu_star)**2),axis=0)
        den = V1 - (V2/V1)

        unbiased_weighted_std = np.sqrt(num/den)
    else:
        unbiased_weighted_std = np.zeros_like(w, dtype=int)

    return unbiased_weighted_std



# def mc_direct(cluster_name,runs, sim_cluster_final):

#     os.chdir("/pool/sao/vpatel/Cluster_Archive/simulated_clusters")
#     threshold_array = np.arange(0, 100500, 500)

#     king = sim_cluster_final.king2d()
#     region_simulation = sim_cluster_final.add_noise(king)
#     final_sim = sim_cluster_final.add_real_error(region_simulation)
    
#     T_obs = np.load(f'/pool/sao/vpatel/Cluster_Archive/{cluster_name}/total_mem.npy')
#     cluster_no = cluster_name.split('_')[1]

#     table = final_sim

#     print(f"Analyzing Cluster {cluster_no}...")

#     cluster_dir = f"/pool/sao/vpatel/Cluster_Archive/simulated_clusters/{cluster_name}_sim"

#     os.makedirs(cluster_dir, exist_ok=True)

#     # Change the current working directory to the newly created directory
#     os.chdir("/pool/sao/vpatel/Cluster_Archive/simulated_clusters/"+f"{cluster_name}_sim")

#     data = table.to_pandas()

#     #==================================================Creating Membership Records===================================
#     # Step 1: Creating membership records.

#     print("Generating membership records...........")
#     print(f"{os.cpu_count()} cores have been put to work! So relax...")  

#     if  os.path.isfile(f"membership_record_100k_{cluster_name}.npy"):
#     # File exists, load the dictionary
#         membership_record_100k = np.load(f"membership_record_100k_{cluster_name}.npy", allow_pickle=True).item()
#         print("100k dictionary loaded successfully.")
#     else:
#         print("100k dictionary does not exist. Saving new dictionary.")
#         # # Step 1: Creating membership records
#         membership_record_100k = {}

#         for i in range(runs):
#             membership_record_100k[i] = run_hdbscan_parallel(data, n=100000, cluster_no = 0, prob_thr = 0.1)
#             print(f'Run: {i+1}. Completed 100000 iterations!')

#         # np.save(f'membership_record_100k_{cluster_name}.npy', membership_record_100k)

#     # ===================================x========================================x=======================x==========       
#     false_pos = []
#     true_mem = []

#     for threshold in threshold_array:

#         # print(f'Current threshold = {threshold}')
#         # #================================HDBSCAN-MC stability check and common member identification=================
#         # print("Performing HDBSCAN-MC stability check and common member identification...........")    


#         #################################################################################################################

#         no_of_iterations = 100000
#         storage_100k = {}

#         for i in range(len(membership_record_100k)):
            
#             # hist, bin_edges = np.histogram(membership_record_100k[i], bins=no_of_bins, range=(0, no_of_iterations), density=True)
            
#             # # Create KDE
#             # kde = gaussian_kde(membership_record_100k[i], bw_method= bw)

#             # # Evaluate the KDE on a grid
#             # x_grid = np.linspace(0, no_of_iterations, 100000)
#             # y_kde = kde(x_grid)

#             ######################################################################################
#             ### Code below is to find MC_thr to detect genuine cluster members
#             MC_thr = threshold
#             ######################################################################################
            
#             # Store data in the dictionary
#             storage_100k[f'iter_{i}'] = data[membership_record_100k[i] > MC_thr]

#         # # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
#         # # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

#         # Get the keys from storage corresponding to iter_1 to iter_10
#         keys = ['iter_' + str(i) for i in range(0, runs)]

#         # Initialize common_members with the 'source_id' column from the first iteration
#         common_members_100k = storage_100k[keys[0]][['source_id']]

#         # Iterate over the keys and merge each 'source_id' column with common_members
#         for key in keys[1:]:
#             common_members_100k = pd.merge(common_members_100k, storage_100k[key][['source_id']], on='source_id', how='inner')

#         ind = data['source_id'].isin(common_members_100k['source_id'])
#         cluster_data = data[ind]

        
#         true_mem.append(np.sum(cluster_data.iloc[:,13]))

#         # print(f"Total false positive detections: {len(cluster_data) - np.sum(cluster_data.iloc[:,13])}")
#         false_pos.append(len(cluster_data) - np.sum(cluster_data.iloc[:,13]))

#     t = np.array(true_mem)
#     f = np.array(false_pos)
#     T = t + f
     
#     residual = np.mean((T_obs - T)**2)
#     print(residual)
#     print("\n")
#     print(sim_cluster_final.center_distance, sim_cluster_final.member_fraction, sim_cluster_final.noise_lower, sim_cluster_final.noise_upper, sim_cluster_final.c, sim_cluster_final.pm_scale, sim_cluster_final.pm_noise_std)

#     del storage_100k , membership_record_100k
#     gc.collect()

#     return residual, np.array(T) , np.array(f), np.array(t)