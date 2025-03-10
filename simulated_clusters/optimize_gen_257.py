import os
import gc
import warnings
import importlib
import numpy as np
import pandas as pd
import configparser

import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.special import kl_div
from scipy.stats import invweibull, gaussian_kde
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

import warnings
from astropy.utils.exceptions import AstropyWarning
# Ignore Astropy warnings
warnings.simplefilter('ignore', category=AstropyWarning)


# Local imports
import HDBSCAN_func
import MS_engine
import cluster_model

# Reload the modules if needed
importlib.reload(HDBSCAN_func)
importlib.reload(MS_engine)
importlib.reload(cluster_model)

from HDBSCAN_func import *
from MS_engine import *
from cluster_model import *




# =================================x=============================x========================================
# # The color palette given below represents muted qualitative colour scheme,that is 
# colour-blind safe with more colours, but lacking a clear red or medium blue.
# Pale grey is meant for bad data in maps.
## Colors are as follows:
# cpalette[0] = Indigo, cpalette[1] = Cyan, cpalette[2] = teal, cpalette[3] = greem, cpalette[4] = olive, 
# cpalette[5] = Sand, cpalette[6] = rose, cpalette[7] = wine, cpalette[8] = purple, cpalette[9] = pale grey
# last color cpalette[10] (dark yellow) added from medium contrast color scheme.
# see https://personal.sron.nl/~pault/

# Initial settings
cpalette = ["#33228b", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677","#882255","#aa4499", "#dddddd","#997700"]
gcolor = ["orange","#18dbcb", "red", "#93f707","#18dbcb", "white", "#f00ea1", "goldenrod", "blue", "sienna", "#357362"] * 20
gmarker = ["o", "p", "s", "v", ">", "<", "p", "s", "d", 'o'] * 20

cluster_name = 'Cluster_257'
radius_factor = 1.5
cluster_no = cluster_name.split('_')[1]
sim_cluster_name = cluster_name
print(f"Analyzing Cluster {cluster_no}")

# User-defined primary directory
primary_dir = config["paths"]["primary_dir"]
sim_dir = config["paths"]["sim_dir"]
sim_data_dir = config["paths"]["sim_data_dir"]

# Construct other directory paths based on the primary directory
cluster_dir = os.path.join(primary_dir, cluster_name)
sim_cluster_dir = os.path.join(sim_dir,f"{cluster_name}_sim")
cluster_statistics_directory = os.path.join(cluster_dir, f"{cluster_name}.txt")

# sim_cluster_dir = os.path.join(sim_dir,f"{cluster_name}_sim")

cluster_statistics = pd.read_table(cluster_statistics_directory, delimiter=',')
kinematic_dist = cluster_statistics[' kinematic_dist'][0]
# total_yso_org = cluster_statistics[' total_HDBSCAN_MC'][0] + cluster_statistics[' total_gaia'][0]
avg_err = cluster_statistics[' avg_dist_unc'][0]
median_distance = cluster_statistics[' median_dist'][0]

real_org = Table.read(sim_data_dir + f'/{cluster_name}_sim/{cluster_name}_region.fits') # Orginal observed data for cluster region without excuding "confirmed" field YSOs
real_org_df = real_org.to_pandas()
tau_aur_gaia = Table.read(sim_data_dir+ f'/more_real_pm_cut_2.fits')

mean_PMRA = -24.145004468499852        # Mean PMRA and PMDEC obtained with 2D KDE approximation
mean_PMDEC = -25.020519831523515

if not os.path.exists(sim_data_dir+ f"/{cluster_name}_sim"):
    # If the directory doesn't exist, create it
    os.makedirs(sim_data_dir+ f"/{cluster_name}_sim")

os.makedirs(cluster_dir, exist_ok=True)
os.chdir(primary_dir)

## Storing the compiled Gaia-SFOG dataset as a astropy table
data_path = "DATA/"
real_data=Table.read(data_path+"SFOG_plus_gaiaYSO_final_dist_XYZ(J2000)_good_astrometry_outer_galaxy.fits")

cluster_detail = Table.read(data_path+"YSO_cluster.fits").to_pandas()
sfog_full_data = Table.read(data_path+"SFOG_catalog_to_use.fits")
sfog_cluster_data = sfog_full_data[sfog_full_data['Cluster']==int(cluster_no)]

# Change to the analyzed cluster directory
os.chdir(f"{cluster_name}/")

cen_RA = round(float(cluster_detail['Central RA'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0]),6)
cen_Dec  = round(float(cluster_detail['Central Dec'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0]),6)
cluster_radius = float(cluster_detail['Circular Radius'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0]) * radius_factor 

# Convert central RA and Dec to SkyCoord object
central_coord = SkyCoord(ra=cen_RA*u.deg, dec=cen_Dec*u.deg, frame='icrs')
t_yso_coords = SkyCoord(ra=real_data['ra_epoch2000_x']*u.deg, dec=real_data['dec_epoch2000_x']*u.deg, frame='icrs')

# Calculate angular separation between each point in ir_yso and central coordinates
angular_separation_t = central_coord.separation(t_yso_coords)

# # Select YSOs within the cluster radius
t = real_data[angular_separation_t < cluster_radius*u.deg]
data = t.to_pandas()

os.chdir(directory+ f"/{cluster_name}_sim")
t.write(f'{cluster_name}_region.fits',format = 'fits',overwrite = True)

# ================================Cluster simulator starts here!===========================================================

class simulate_cluster:
    """
    A class to simulate a star cluster.

    Attributes:
        center_distance (float): The distance to the center of the cluster.
        member_fraction (float): The fraction of members in the cluster.
        noise_lower (float): The distance lower bound for field ysos.
        noise_upper (float):  The distance upper bound for field ysos.
        c (float): Compactness fraction.
        pm_scale (float): Proper motion scaling parameter.
        cluster_name (str): The name of the cluster.
        center_ra (float): The right ascension of the cluster center.
        center_dec (float): The declination of the cluster center.
    """
     
    def __init__(self, center_distance = None, member_fraction = None, noise_lower = None, noise_upper = None, c = None, pm_scale = None, 
                 cluster_name = sim_cluster_name, 
                 center_ra = cen_RA, 
                 center_dec = cen_Dec):
        
        self.cluster_name = cluster_name
        self.center_distance = center_distance
        self.center_ra = center_ra * u.deg
        self.center_dec = center_dec * u.deg
        if member_fraction is None:
            self.member_fraction = member_fraction
        else:
            self.member_fraction = member_fraction/100
        self.pm_scale = pm_scale/1000

        self.noise_lower = noise_lower
        self.noise_upper = noise_upper
        if c is None:
            self.c = c
        else:
            self.c = c/1000

        if self.member_fraction is not None:
            self.num_stars = round(self.member_fraction*total_yso)
            self.noise_ysos = total_yso - self.num_stars

        cluster_radius_deg =  float(cluster_detail['Circular Radius'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0])
        if self.center_distance is not None:
            self.r_tide = np.tan(np.deg2rad(cluster_radius_deg))* self.center_distance * u.pc
            self.r_core = self.r_tide*self.c # c is compactness fraction  

    def assign(self, mem_frac, c):
        if self.member_fraction is not None:
            self.member_fraction = mem_frac/100
            self.num_stars = round(self.member_fraction*total_yso)
            self.noise_ysos = total_yso - self.num_stars

        if self.c is not None:
            self.c = c/1000

        cluster_radius_deg =  float(cluster_detail['Circular Radius'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0])
        if self.center_distance is not None:
            self.r_tide = np.tan(np.deg2rad(cluster_radius_deg))* self.center_distance * u.pc
            self.r_core = self.r_tide*self.c # c is compactness fraction  

#============================================Using cluster models======================================================
        
    # # Wrapper functions for cluster models
    # def uniform(self):
    #     return uniform_cluster(self)

    # def gaussian(self):
    #     return gaussian_cluster(self)
    
    def king2d(self,Wpermit=0):
        king = King2D(self, kde_pmra, kde_pmdec, kde_parallax, Wpermit)
        return king
        
##================================================================ Code for generating noise ================================================
    
    def add_noise(self, king, Wpermit = 0):
        region_simulation = generate_noise(self, king, Wpermit)
        return region_simulation
## ===================================================== Code for error extraction ===========================================    
    def add_real_error(self, region_simulation, real_data = real_data, Wpermit = 0, update_permit = 0):
        sim = real_error(self,region_simulation, real_data, Wpermit, update_permit, dist_limits , PMRA_limits, PMDEC_limits, mean_PMRA/100, mean_PMDEC/100)
        return sim
    
#==================================x========================================x=======================x==========
# Extract the necessary data
d_16_field = real_org['final_dist_lo']
d_50_field = real_org['final_dist']
d_84_field = real_org['final_dist_hi']

# Compute k and λ in a vectorized way
k_field = np.log(np.log(0.84) / np.log(0.16)) / np.log(d_16_field / d_84_field)
λ_field = d_50_field * np.log(2)**(1/k_field)

# Set random seed
np.random.seed(6)

# Generate the distributions and calculate the 3-sigma bounds
inv_distributions = [invweibull.rvs(c=k, scale=λ, size=10**5) for k, λ in zip(k_field, λ_field)]
dist_3sigma_upper = [np.percentile(dist, 99.865) for dist in inv_distributions]
dist_3sigma_lower = [np.percentile(dist, 0.135) for dist in inv_distributions]
dist_limits_org = [dist_3sigma_lower, dist_3sigma_upper]

PMRA_3sigma_upper = np.percentile(real_org['pmra'], 99.865)
PMRA_3sigma_lower = np.percentile(real_org['pmra'], 0.135)
PMRA_limits_org = [PMRA_3sigma_lower, PMRA_3sigma_upper]

PMDEC_3sigma_upper = np.percentile(real_org['pmdec'], 99.865)
PMDEC_3sigma_lower = np.percentile(real_org['pmdec'], 0.135)
PMDEC_limits_org = [PMDEC_3sigma_lower, PMDEC_3sigma_upper]

sim_cluster = simulate_cluster(pm_scale = 1000)  # An arbitrary value of pm_scale is used for 
                                                 # the initialization of the class.  It 
                                                 # will be updated to the optimized value in
                                                 # proper motion optimization stage.

cluster_radius_deg =  float(cluster_detail['Circular Radius'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0])

# Fit the KDE to the Taurus-Auriga (TA) complex data so that additional data can be generated from the 
# aprroximated KDEs of pmra, pmdec and parallax of TA region
# These are then appropriately scaled to the analyzed cluster distance
# by pm_scale parameter in proper motion optimization stage.
kde_pmra = gaussian_kde(tau_aur_gaia['pmra'], bw_method = None)
kde_pmdec = gaussian_kde(tau_aur_gaia['pmdec'], bw_method = None)
kde_parallax = gaussian_kde(tau_aur_gaia['parallax'], bw_method = None)

def outlier_detect_observed(center_dist_limits, cluster_radius_deg, cluster_real, dist_limits , PMRA_limits, PMDEC_limits):
    """
    Detect outliers in the observed cluster data based on distance and proper motion limits.

    Parameters:
    center_dist_limits (list): A list containing the lower and upper limits of the cluster center distance.
    cluster_radius_deg (float): The radius of the cluster in degrees.
    cluster_real (astropy.table.Table): The observed cluster data.
    dist_limits (list): A list containing the lower and upper 3 sigma limits of the YSO distances.
    PMRA_limits (list): A list containing the lower and upper 3 sigma limits of the YSO proper motions in RA.
    PMDEC_limits (list): A list containing the lower and upper 3 sigma limits of the YSO proper motions in DEC.

    Returns:
    astropy.table.Table: A table containing the outliers in the observed cluster data.
    """

    r_tide_upper = np.tan(np.deg2rad(cluster_radius_deg))* center_dist_limits[1]
    r_tide_lower = np.tan(np.deg2rad(cluster_radius_deg))* center_dist_limits[0]
    # Including confirmed outliers by replacing closest members based on median distances
    cluster_upper = center_dist_limits[1] + r_tide_upper
    cluster_lower = center_dist_limits[0] - r_tide_lower

    mask_nonmem = (
        (cluster_upper < dist_limits[0]) | 
        (cluster_lower > dist_limits[1]) | 
        (cluster_real['pmra'] < PMRA_limits[0]) | 
        (cluster_real['pmra'] > PMRA_limits[1]) | 
        (cluster_real['pmdec'] < PMDEC_limits[0]) | 
        (cluster_real['pmdec'] > PMDEC_limits[1])
    )
    outliers_cluster_observed = cluster_real[mask_nonmem]

    return outliers_cluster_observed

# Exclude the outliers from the observed cluster data
# by assuming that the cluster center distance is within 
# 20% of the kinematic distance estimate
center_dist_limits_max = [0.8*kinematic_dist, 1.2*kinematic_dist]

excluded_ysos = outlier_detect_observed(center_dist_limits_max, cluster_radius_deg, real_org, dist_limits_org , PMRA_limits_org, PMDEC_limits_org).to_pandas()
outer = real_org_df['source_id'].isin(excluded_ysos['source_id'])
real = Table.from_pandas(real_org_df[~outer])
real.write(f'{cluster_name}_region_cleaned.fits',format = 'fits',overwrite = True) #cluster region files after excuding "confirmed" field YSOs
total_yso = len(real)

print(f'The number of excluded YSOs are: {np.sum(outer)}')
print(f'The number of YSOs analyzed in the observed cluster region are: {total_yso}')

# "data_updated" is the reifned YSO collection which removes the confirmed field YSOs and the outliers
# from the observed cluster data. This observed data is used for further cluster membership analysis
data_updated = Table.read(sim_data_dir+ f"/{cluster_name}_sim/{cluster_name}_region_cleaned.fits").to_pandas()

#==================================================Creating Membership Records===================================
# Creating membership records.

print("Generating membership records...........")
print(f"{os.cpu_count()} cores have been put to work! So relax...")  
threshold_array = np.arange(0, 100500, 500)

if  os.path.isfile(cluster_dir+f"/membership_record_100k_{cluster_name}.npy"):
# File exists, load the dictionary
    membership_record_100k = np.load(cluster_dir + f"/membership_record_100k_{cluster_name}.npy", allow_pickle=True).item()
    print("100k dictionary loaded successfully.")
else:
    print("100k dictionary does not exist. Saving new dictionary.")
    # Creating membership records for 100k iterations
    membership_record_100k = {}

    for i in range(10):
        membership_record_100k[i] = run_hdbscan_parallel(data_updated, n=100000, cluster_no = 0, prob_thr = 0.1)
        print(f'Run: {i+1}. Completed 100000 iterations!')

    np.save(cluster_dir + f"/membership_record_100k_{cluster_name}.npy", membership_record_100k)

if os.path.isfile(cluster_dir+f"/membership_record_1k_{cluster_name}.npy"):
    # File exists, load the dictionary 
    membership_record_1k = np.load(cluster_dir+f"/membership_record_1k_{cluster_name}.npy", allow_pickle=True).item()
    print("1k dictionary loaded successfully.")
else:
    print("1k dictionary does not exist. Saving new dictionary.")
    # Creating membership records for 1000 iterations

    membership_record_1k = {}

    for i in range(10):
        membership_record_1k[i] = run_hdbscan_parallel(data_updated, n=1000, cluster_no = 0, prob_thr = 0.1)
        print(f'Run: {i+1}. Now completed 1000 iterations.')

    np.save(cluster_dir+f"/membership_record_1k_{cluster_name}.npy", membership_record_1k)


# ===================================x========================================x=======================x==========    
# This part of code calculates the total number of YSOs detected in the cluster region as the Monte Carlo threshold is varied from 0 to 100,000.
# The total number of YSOs detected is stored in an array named total_mem and saved as a numpy file.
# The array is then used to plot the variation of true members and false positive detections for the observed cluster.
# i.e. Monte Carlo spectra for the observed cluster.   
total_mem = []

for threshold in threshold_array:
  
    annotation_coords = (0.02, 0.95)
    y = []
    no_of_bins = 100

    no_of_iterations = 100000
    storage_100k = {}

    for i in range(len(membership_record_100k)):
        
        MC_thr = threshold
        
        # Store data in the dictionary
        storage_100k[f'iter_{i}'] = data_updated[membership_record_100k[i] > MC_thr]

    # # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
    # # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

    # Get the keys from storage corresponding to iter_1 to iter_10
    keys = ['iter_' + str(i) for i in range(0, 10)]

    # Initialize common_members with the 'source_id' column from the first iteration
    common_members_100k = storage_100k[keys[0]][['source_id']]

    # Iterate over the keys and merge each 'source_id' column with common_members
    for key in keys[1:]:
        common_members_100k = pd.merge(common_members_100k, storage_100k[key][['source_id']], on='source_id', how='inner')

    # # np.save('true_mem.npy',np.array(true_mem))
    # # np.save('false_neg.npy',np.array(false_neg))

    ind = data_updated['source_id'].isin(common_members_100k['source_id'])
    cluster_data = data_updated[ind]

    total_mem.append(len(cluster_data))
        
np.save(cluster_dir+f'/total_mem.npy',np.array(total_mem))

## Plotting the variation of true members and false positive detections for observed/real cluster 
## In other words this code block generates Monte Carlo spectra for the observed cluster

# Set dark background style
plt.style.use('dark_background')

# Assuming true_mem and false_pos are already defined as numpy arrays
x_axis = threshold_array

fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

# Calculate the range of the data
y_range = max(total_mem) - min(total_mem)

# Determine the appropriate tick interval based on the range
if y_range <= 200:
    major_tick_interval = 20
    minor_tick_interval = 10
elif y_range <= 500:
    major_tick_interval = 50
    minor_tick_interval = 25
else:
    major_tick_interval = 100
    minor_tick_interval = 50

fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
ax.plot(x_axis, total_mem, color='orange', label='Total Detections')

# Setting y-axis limits with a slight padding
ax.set_ylim([0, max(total_mem) * 1.05])
ax.set_xlim([0, max(x_axis) * 1.02])

# Setting tick locators
ax.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
ax.xaxis.set_major_locator(MultipleLocator(10000))  # Set major ticks every 10000 units
ax.xaxis.set_minor_locator(MultipleLocator(1000))   # Set minor ticks every 1000 units

# Setting tick parameters
ax.tick_params(axis='both', which='minor', direction='in', width=0.75, length=4, bottom=True, top=True, left=True, right=True)
ax.tick_params(axis='both', which='major', direction='in', width=1, length=9, bottom=True, top=True, left=True, right=True,labelsize=8)
# ax.set_yscale('log')
# ax.set_xscale('log')

ax.legend(fontsize=13)

# Adding labels and title
ax.set_xlabel('Monte Carlo Threshold',fontsize = 15)
ax.set_ylabel('No of YSOs',fontsize = 15)
ax.set_title(f'Variation of true members and false positive detections for real {cluster_name}',fontsize = 17)

# Set grid lines thinner and fainter
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# ax1 = ax.twinx()
# ax.plot(x_axis, true_mem, label='True Members')

# Adjusting layout
plt.tight_layout()

plot_filename_mspec = cluster_dir+f"/mspec_real_{cluster_name}.png"
plt.savefig(plot_filename_mspec, dpi=300)

plt.close()

del membership_record_100k
del total_mem
gc.collect()

#==================================x=============================================================x==================================
## The spatial optimization setup starts here

lmin = np.min(real['final_dist_lo'])
lmax = np.percentile(real['final_dist_hi'],2.5)

umin = np.percentile(real['final_dist_lo'],97.5)
umax = np.percentile(real['final_dist_hi'],99.85)

PMRA = np.array(real['pmra']) 
PMDEC = np.array(real['pmdec'])


# Defining bounds for the optimization parameters
bounds = [(center_dist_limits_max[0], center_dist_limits_max[1]), (0.5*100, 0.99*100), (lmin, lmax), (umin, umax), (0.01*1000, 0.99*1000)]

X = np.array(real['X'])
Y = np.array(real['Y'])
Z = np.array(real['Z'])

bin_factor = 5
def execute(params, X, Y, Z, sim_cluster):
    """
    Execute the optimization process for the simulated cluster.

    Parameters:
    params (list): A list of parameters for the simulation. 
                   [center_distance, member_fraction, noise_lower, noise_upper, c]
    X (numpy.ndarray): Array of X coordinates of the observed data.
    Y (numpy.ndarray): Array of Y coordinates of the observed data.
    Z (numpy.ndarray): Array of Z coordinates of the observed data.
    sim_cluster (simulate_cluster): An instance of the simulate_cluster class.

    Returns:
    float: The total KL divergence between the probability distribution of observed and simulated YSO's 
           x, y and z distance 
    """
    sim_cluster.center_distance =params[0]
    sim_cluster.member_fraction = params[1]
    sim_cluster.noise_lower = params[2]
    sim_cluster.noise_upper  = params[3]
    sim_cluster.c = params[4]

    sim_cluster.assign(sim_cluster.member_fraction, sim_cluster.c)
    king = sim_cluster.king2d()
    region_simulation = sim_cluster.add_noise(king)
    sim = sim_cluster.add_real_error(region_simulation, update_permit = 1)

    x = np.array(sim['X'])
    y = np.array(sim['Y'])
    z = np.array(sim['Z'])

    # Compute histograms for X, Y, and Z
    hist_X, _ = np.histogram(X, bins=round(len(X)/bin_factor), density=False)
    hist_x, _ = np.histogram(x, bins=round(len(x)/bin_factor), density=False)
    hist_Y, _ = np.histogram(Y, bins=round(len(Y)/bin_factor), density=False)
    hist_y, _ = np.histogram(y, bins=round(len(y)/bin_factor), density=False)
    hist_Z, _ = np.histogram(Z, bins=round(len(Z)/bin_factor), density=False)
    hist_z, _ = np.histogram(z, bins=round(len(z)/bin_factor), density=False)


    # Normalize histograms to obtain probability distributions
    prob_X = (hist_X / np.sum(hist_X)) 
    prob_x = (hist_x / np.sum(hist_x)) + 10**-9
    prob_Y = (hist_Y / np.sum(hist_Y)) 
    prob_y = (hist_y / np.sum(hist_y)) + 10**-9
    prob_Z = (hist_Z / np.sum(hist_Z)) 
    prob_z = (hist_z / np.sum(hist_z)) + 10**-9

    # Calculate KL divergence
    KL_X = np.sum(kl_div(prob_X, prob_x))
    KL_Y = np.sum(kl_div(prob_Y, prob_y))
    KL_Z = np.sum(kl_div(prob_Z, prob_z))

    total_kl_divergence = KL_X + KL_Y + KL_Z

    return total_kl_divergence 

# Extract the necessary data
d_16_field = real['final_dist_lo']
d_50_field = real['final_dist']
d_84_field = real['final_dist_hi']

# Compute k and λ in a vectorized way
k_field = np.log(np.log(0.84) / np.log(0.16)) / np.log(d_16_field / d_84_field)
λ_field = d_50_field * np.log(2)**(1/k_field)

# Set random seed
np.random.seed(6)

# Generate the distributions and calculate the 3-sigma bounds
inv_distributions = [invweibull.rvs(c=k, scale=λ, size=10**5) for k, λ in zip(k_field, λ_field)]
dist_3sigma_upper = [np.percentile(dist, 99.865) for dist in inv_distributions]
dist_3sigma_lower = [np.percentile(dist, 0.135) for dist in inv_distributions]
dist_limits = [dist_3sigma_lower, dist_3sigma_upper]

PMRA_3sigma_upper = np.percentile(real['pmra'], 99.865)
PMRA_3sigma_lower = np.percentile(real['pmra'], 0.135)
PMRA_limits = [PMRA_3sigma_lower, PMRA_3sigma_upper]

PMDEC_3sigma_upper = np.percentile(real['pmdec'], 99.865)
PMDEC_3sigma_lower = np.percentile(real['pmdec'], 0.135)
PMDEC_limits = [PMDEC_3sigma_lower, PMDEC_3sigma_upper]

def outlier_detect(center_dist, total_yso, cluster_radius_deg, cluster_real, dist_limits , PMRA_limits, PMDEC_limits):
    """
    Detect outliers in the observed cluster data based on distance and proper motion limits.  

    The function returns the fraction of outliers in the observed cluster data corresponding 
    to the given **center_dist** parameter in the simulation. The tidal radius of the cluster 
    at this **center_dist** value defines the distance limits (**dist_limits**).  

    Thus, it establishes a relationship between **mem_frac** and the **center_dist** 
    parameter within the constraint function during the spatial optimization process.

    Parameters:
    center_dist (float): The estimated center distance of the cluster.
    total_yso (int): The total number of young stellar objects (YSOs) in the cluster.
    cluster_radius_deg (float): The radius of the cluster in degrees.
    cluster_real (astropy.table.Table): The observed cluster data after removing initial confirmed field YSOs.
    dist_limits (list): A list containing the lower and upper limits of the YSO distances.
    PMRA_limits (list): A list containing the lower and upper limits of the YSO proper motions in RA.
    PMDEC_limits (list): A list containing the lower and upper limits of the YSO proper motions in DEC.

    Returns:
    float: The fraction of outliers in the observed cluster data.
    """
    r_tide = np.tan(np.deg2rad(cluster_radius_deg))* center_dist

    cluster_upper = center_dist + r_tide
    cluster_lower = center_dist - r_tide

    mask_nonmem = (
        (cluster_upper < dist_limits[0]) | 
        (cluster_lower > dist_limits[1]) | 
        (cluster_real['pmra'] < PMRA_limits[0]) | 
        (cluster_real['pmra'] > PMRA_limits[1]) | 
        (cluster_real['pmdec'] < PMDEC_limits[0]) | 
        (cluster_real['pmdec'] > PMDEC_limits[1])
    )
    outliers_cluster_real = cluster_real[mask_nonmem]

    return len(outliers_cluster_real)/total_yso  

# Constraint function that applies only to the first two parameters i.e center_dist and mem_frac
def constraint_function(params):
    center_dist, mem_frac, noise_lower, noise_upper = params[0], params[1]/100, params[2], params[3]
    return mem_frac - (1 - outlier_detect(center_dist, total_yso, cluster_radius_deg, real, dist_limits , PMRA_limits, PMDEC_limits)), center_dist - noise_upper, center_dist - noise_lower

# Define the constraint as a NonlinearConstraint
nlc = NonlinearConstraint(constraint_function, [-np.inf, -np.inf, 0], [0, 0, np.inf])

num_runs = 60

best_fit_object = []  # List to store the best-fit results
best_res = None # Initialize the best result to None

# Run the spatial optimization multiple times
print('Starting differential evolution for spatial optimization!\n')

#**************************************************************************************************
## Comment the lines below if using stored values for 
## best_fit_params and best_fit_params_pm

for x in range(num_runs):   
    print(f'Spatial optimization run {x}!')
    # Perform optimization using differential evolution with constraints
    res = differential_evolution(execute, bounds, constraints = nlc, args=(X, Y, Z, sim_cluster), popsize = 150, seed = x, mutation=(0.5, 1.5), recombination = 0.5, strategy='best1bin', tol= 0.5, integrality = [1,1,1,1,1], polish = False, disp = False, workers = 80) 
    # Print the results 
    print('Best-fit parameters: ', res.x)
    print('Best-fit objective value:', res.fun)
    best_fit_object.append(res)


sorted_indices = np.argsort([obj.fun for obj in best_fit_object])   # Use np.argsort to get the indices that would sort the array
best_fit_params = np.array([best_fit_object[i].x for i in sorted_indices])  # Reorder the best_fit_objects_list based on the sorted indices
np.save('best_fit_spatial.npy', best_fit_params)
#**************************************************************************************************

# top_num = 30
# os.chdir(directory+ f"/{cluster_name}_sim")
## best_fit_params = np.load('best_fit_spatial.npy')[0:top_num]
## best_fit_params_pm = np.load('best_fit_pm.npy')
## print(best_fit_params_pm)
## T_good = np.load(f'T_good_{cluster_name}.npy').tolist()
## residual = np.load(f'residual_{cluster_name}.npy')
## print(best_fit_params)

print("============= Completed spatial optimization! Starting proper motion optimization....!========================")


def execute_pm(params_pm, PMRA, PMDEC, sim_cluster):
    """
    Execute the proper motion optimization process for the simulated cluster.

    Parameters:
    params_pm (list): A list of parameters for the simulation. 
                      [pm_scale]
    PMRA (numpy.ndarray): Array of proper motion in RA of the observed data.
    PMDEC (numpy.ndarray): Array of proper motion in DEC of the observed data.
    sim_cluster (simulate_cluster): An instance of the simulate_cluster class.

    Returns:
    float: The total KL divergence between the probability distribution of observed and simulated YSO's 
           proper motion in RA and DEC.
    """
    sim_cluster.pm_scale = params_pm[0]/1000

    king = sim_cluster.king2d()
    region_simulation = sim_cluster.add_noise(king)
    sim = sim_cluster.add_real_error(region_simulation)

    pmra = np.array(sim['pmra'])
    pmdec = np.array(sim['pmdec'])

    # Compute histograms for X, Y, and Z
    hist_PMRA, _ = np.histogram(PMRA, bins=round(len(PMRA)/bin_factor), density=False)
    hist_pmra, _ = np.histogram(pmra, bins=round(len(pmra)/bin_factor), density=False)
    hist_PMDEC, _ = np.histogram(PMDEC, bins=round(len(PMDEC)/bin_factor), density=False)
    hist_pmdec, _ = np.histogram(pmdec, bins=round(len(pmdec)/bin_factor), density=False)

    # Normalize histograms to obtain probability distributions
    prob_PMRA = (hist_PMRA / np.sum(hist_PMRA)) 
    prob_pmra = (hist_pmra / np.sum(hist_pmra)) + 10**-9
    prob_PMDEC = (hist_PMDEC / np.sum(hist_PMDEC)) 
    prob_pmdec = (hist_pmdec / np.sum(hist_pmdec)) + 10**-9

    # Calculate KL divergence
    KL_PMRA = np.sum(kl_div(prob_PMRA, prob_pmra))
    KL_PMDEC = np.sum(kl_div(prob_PMDEC, prob_pmdec))

    total_kl_divergence_pm = KL_PMRA + KL_PMDEC

    return total_kl_divergence_pm

bounds_pm = [(0.1*1000,9.9*1000)]   # Bounds for the pm_scale parameter

#**************************************************************************************************
## Comment the lines below if using stored values for 
## best_fit_params and best_fit_params_pm

top_num = 30    # Number of best-fit parameters to consider for proper motion optimization
best_fit_params_pm = []
best_fit_params = best_fit_params[0:top_num]
residual = np.empty(0)
T_good = []
f_good = []
t_good = []
mem_record_good = []

for i in range(top_num):

    sim_cluster_spatial = simulate_cluster(center_distance = best_fit_params[i][0], member_fraction = best_fit_params[i][1], noise_lower = best_fit_params[i][2], noise_upper = best_fit_params[i][3], c = best_fit_params[i][4], pm_scale = 1* 1000)

    res_pm = differential_evolution(execute_pm, bounds_pm, args=(PMRA, PMDEC, sim_cluster_spatial), popsize=150, mutation=(0.5, 1.5), recombination = 0.5,  strategy='best1bin', integrality = [1], seed = i, tol = 0.5, workers = 80, polish=False, disp=False)
    print(f'Best-fit parameters: {res_pm.x}')
    print(f'Best-fit objective value: {res_pm.fun}\n')
    best_fit_params_pm.append(res_pm.x)

np.save('best_fit_pm.npy',best_fit_params_pm)
# best_fit_params_pm = np.load('best_fit_pm.npy')

for i in range(top_num):
    sim_cluster_final = simulate_cluster(center_distance = best_fit_params[i][0], member_fraction = best_fit_params[i][1], noise_lower = best_fit_params[i][2], noise_upper = best_fit_params[i][3], c = best_fit_params[i][4], pm_scale = best_fit_params_pm[i][0])

    residual_new,T_new,f_new,t_new, membership_record_100k = monte_carlo_spectra(cluster_name,10, sim_cluster_final)
    residual = np.append(residual,residual_new)
    T_good.append(T_new)
    t_good.append(t_new)
    f_good.append(f_new)
    mem_record_good.append(membership_record_100k)

#**************************************************************************************************

#**************************************************************************************************
## Uncomment the code below if the stored optimization results need to be used for further analysis
# top_num = 30
# os.chdir(sim_data_dir+ f"/{cluster_name}_sim")
# best_fit_params = np.load('best_fit_spatial.npy')[0:top_num]
# best_fit_params_pm = np.load('best_fit_pm.npy')
# T_good = np.load(f'T_good_{cluster_name}.npy').tolist()
# t_good = np.load(f't_good_{cluster_name}.npy').tolist()
# f_good = np.load(f'f_good_{cluster_name}.npy').tolist()
# residual = np.load(f'residual_{cluster_name}.npy')

# os.makedirs(sim_cluster_dir, exist_ok=True)
# ms_fit_membership_record_100k = np.load(sim_cluster_dir + f'/membership_record_100k_{cluster_name}.npy', allow_pickle=True)
# print(residual)
# print("\n")

# print("Required optimization results and ms spectrum [f,t,T] variables loaded successfully!")
#**************************************************************************************************

# Find the index of the best fit parameters with the minimum residual 
# between observed and simulated Monte Carlo spectra
best_fit_index = np.argmin(residual) 

# Extract the best fit parameters for spatial and proper motion optimization
ms_fit_params = best_fit_params[best_fit_index]
ms_fit_params_pm = best_fit_params_pm[best_fit_index]

# Extract the corresponding T, t, and f values for the best fit for best fit model
ms_fit_T = T_good[best_fit_index] # T_good is the total YSO variation with Montecarlo theshold for best fit model
ms_fit_t = t_good[best_fit_index] # f_good is the variation of false-positives with Montecarlo theshold for best fit model
ms_fit_f = f_good[best_fit_index] # t_good is the variation of true-positives with Montecarlo theshold for best fit model

#****************************************************************************
## Comment the lines below if using stored values for 
## best_fit_params and best_fit_params_pm

# Extract the membership record for the best fit
ms_fit_membership_record_100k = mem_record_good[best_fit_index]
best_fit_params_pm =  np.array(best_fit_params_pm)

os.makedirs(sim_cluster_dir, exist_ok=True)
np.save(sim_cluster_dir + f'/membership_record_100k_{cluster_name}.npy', ms_fit_membership_record_100k)

os.chdir(sim_data_dir+ f"/{cluster_name}_sim")

# Save the best fit parameters and other results to a numpy file
np.save('best_fit_pm.npy',best_fit_params_pm)
np.save(f'residual_{cluster_name}.npy',residual)
np.save(f'T_good_{cluster_name}.npy', np.array(T_good))
np.save(f't_good_{cluster_name}.npy', np.array(t_good))
np.save(f'f_good_{cluster_name}.npy', np.array(f_good))
#****************************************************************************

# Convert residuals to float and create a copy for printing
residual = residual.astype(float)
residual_print = residual.copy()

# Define a cut-off value for residuals
cut_off = round(3*min(residual_print))
mask = (residual <= cut_off) # Create a mask for residuals below the cut-off value

# Filter T_good and f_good based on the mask
T_good = np.array([T for T, m in zip(T_good, mask.flatten()) if m])
f_good = np.array([f for f, m in zip(f_good, mask.flatten()) if m])

# Set residuals above the cut-off value to infinity
# so that weights of such models is zero
residual[(residual > cut_off)] = np.inf 

# Calculate weights as the inverse of residuals
weights = 1/residual
print(weights)

# Calculate weighted average and standard deviation of the best fit parameters 
# for spatial optimization
weighted_ms_params = np.average(best_fit_params[0:top_num], axis=0, weights = weights)
std_params = weighted_std_unbiased(best_fit_params[0:top_num], weights)

# Calculate weighted average and standard deviation of the best fit parameters 
# for proper motion optimization
weighted_ms_params_pm = np.average(best_fit_params_pm[0:top_num], axis=0, weights = weights)
std_params_pm = weighted_std_unbiased(best_fit_params_pm[0:top_num], weights)

# Create the final simulated cluster with the best fit parameters
sim_cluster_final = simulate_cluster(
    center_distance=ms_fit_params[0], 
    member_fraction=ms_fit_params[1], 
    noise_lower=ms_fit_params[2], 
    noise_upper=ms_fit_params[3], 
    c=ms_fit_params[4], 
    pm_scale=ms_fit_params_pm[0]
)

# Generate simulated cluster corresponding to the best fit parameters
king = sim_cluster_final.king2d(Wpermit = 1)
region_simulation = sim_cluster_final.add_noise(king, Wpermit = 1)
final_sim = sim_cluster_final.add_real_error(region_simulation, Wpermit = 1, update_permit = 1)

# residual_final,T_final,f_final,t_final, final_membership_record_100k = monte_carlo_spectra(cluster_name,10, sim_cluster_final)
# print(f'The final MSD is: {residual_final}')

# cluster_dir = f"/pool/sao/vpatel/Cluster_Archive/simulated_clusters/{cluster_name}_sim"
# os.makedirs(cluster_dir, exist_ok=True)
# np.save(cluster_dir + f'/membership_record_100k_{cluster_name}.npy', final_membership_record_100k)

# Pad the residual array with zeros to match the desired length in saved text file
padded_residual = np.pad(residual_print, (2, len(best_fit_params) - len(residual_print)), mode='constant', constant_values=-1)

# Create dictionary for saving as text file
data = {
    'center_dist': [weighted_ms_params[0]] + [std_params[0]] + best_fit_params[:, 0].tolist(),
    'mem_frac': [weighted_ms_params[1]] + [std_params[1]] + best_fit_params[:, 1].tolist(),
    'noise_lower': [weighted_ms_params[2]] + [std_params[2]] + best_fit_params[:, 2].tolist(),
    'noise_upper': [weighted_ms_params[3]] + [std_params[3]] + best_fit_params[:, 3].tolist(),
    'c': [weighted_ms_params[4]] + [std_params[4]] + best_fit_params[:, 4].tolist(),
    'pm_scale': [weighted_ms_params_pm[0]] + [std_params_pm[0]] + best_fit_params_pm[:, 0].tolist(),
    'residual': padded_residual.tolist()  
}

df = pd.DataFrame(data)
df.to_csv(f'parameters_{cluster_name}.txt', sep='\t', index=True)

## ===============================Optimization ends here. Generating Monte Carlo spectrum..... Hang tight!==============================
print("Optimization ends here! Generating Monte Carlo spectrum..... Hang tight!")

os.chdir(sim_dir)

threshold_array = np.arange(0, 100500, 500)

clusters = [cluster_no]
radius_factor = [radius_factor]
runs = 10

for cluster,rf in zip(clusters,radius_factor):

    from astropy.table import Table,join

    os.chdir(sim_dir)

    cluster_no = cluster
    cluster_name = f'Cluster_{cluster_no}'
    print(f"Analyzing Cluster {cluster_no}...")

    data_path = f"Data/{cluster_name}_sim"
    table=Table.read(data_path+f"/required_sim_data_{cluster_name}.fits")

    os.makedirs(sim_cluster_dir, exist_ok=True)
    os.chdir(f"{cluster_name}_sim")

    data = table.to_pandas()

##====================================================Creating Membership Records for 1000 iterations===================================

    if os.path.isfile(f"membership_record_1k_{cluster_name}.npy"):

        membership_record_1k = np.load(f"membership_record_1k_{cluster_name}.npy", allow_pickle=True).item()
        print("1k dictionary loaded successfully.")
    else:
        print("1k dictionary does not exist. Saving new dictionary.")

        membership_record_1k = {}

        for i in range(runs):
            membership_record_1k[i] = run_hdbscan_parallel(data, n=1000, cluster_no = 0, prob_thr = 0.1)
            print(f'Run: {i+1}. Now completed 1000 iterations.')

        np.save(f'membership_record_1k_{cluster_name}.npy', membership_record_1k)

t = np.array(ms_fit_t)
f = np.array(ms_fit_f)
# t = np.array(t_final)
# f = np.array(f_final)
T = t + f
T_obs = np.load(os.path.join(primary_dir, f"{cluster_name}/total_mem.npy"))
residual = (T_obs - T)

np.save('true_mem.npy',t)
np.save('false_pos.npy',f)

x_axis = threshold_array

# Find the index where t becomes 0.95 * t
MC_thr_ind = np.argmax(t <= 0.95 * t[0])
MC_thr = x_axis[MC_thr_ind]
f_MC_thr = f[MC_thr_ind]
total_in_field = T_obs[MC_thr_ind]
true_MC_thr = T_obs[MC_thr_ind] - f_MC_thr

print(f"The Monte-Carlo threshold is: {MC_thr}")

# Create a dictionary
data = {
    'MC_thr': [MC_thr],
    'f_MC_thr': [f_MC_thr],
    'min_f': [np.min(f_good[:,MC_thr_ind])],
    'max_f': [np.max(f_good[:,MC_thr_ind])],
    'total_in_field': [total_in_field],
    'true_MC_thr': [true_MC_thr],
    'max_t': [T_obs[MC_thr_ind] - np.min(f_good[:,MC_thr_ind])],
    'min_t': [T_obs[MC_thr_ind] - np.max(f_good[:,MC_thr_ind])],
}

df = pd.DataFrame(data)
df.to_csv(f'MC_thr_{cluster_name}.txt', sep='\t', index=False)

print(f"DataFrame saved to parameters_{cluster_name}.txt")

#*******************************************xx**************************************************************************x
## Plotting the Monte Carlo spectra for the observed cluster and simulted cluster models.
## The Monte Carlo spectra for the observed cluster is plotted along with the Monte Carlo spectra for the best fit model.
## Additionally models having mean squared deviation (MSD) less than the cut-off value = 3 x the minimum MSD are also plotted.
#*******************************************xx**************************************************************************x

plt.style.use('default')

# Create a figure with two subplots one for observed and simulated Monte Carlo spectra
# and other for the residual plot.
fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1],'hspace': 0.0, 'wspace': 0.0})

# Calculate the range of the data
y_range = max(T) - min(T)

# Determine the appropriate tick interval based on the range
if y_range <= 200:
    major_tick_interval = 20
    minor_tick_interval = 10
elif y_range <= 500:
    major_tick_interval = 50
    minor_tick_interval = 25
else:
    major_tick_interval = 100
    minor_tick_interval = 50

# Plot only the first column of T_good.T 
ax.plot(x_axis, T_good.T[:, 0], label=f'[1] Simulations with MSD < {cut_off}', color=cpalette[9], zorder = 2)
# Add a vertical line at index_95
ax.plot(x_axis, t, label='[2] Simulated true members', color = cpalette[0], linewidth = 2, linestyle = (0,(3,1,1,1)), zorder = 3)
ax.plot(x_axis, f, label='[3] Simulated false positives', color = cpalette[6], zorder = 2)
ax.axvline(x=MC_thr, color=cpalette[10], linestyle='--',linewidth = 2, label='[4] Monte-carlo threshold', zorder = 4)

# Plot the remaining columns of T_good.T without a label
for i in range(1, T_good.shape[0]):
    ax.plot(x_axis, T_good.T[:, i], color=cpalette[9])

ax.plot(x_axis, T, color ='black', label='[5] Simulated total detections')
ax.plot(x_axis, T_obs , color = cpalette[3], label='[6] Total observed detections',alpha = 0.6, linewidth = 3)

# Set y-axis limits with a slight padding
ax.set_ylim([0, max(T) * 1.05])
ax.set_xlim([0, max(x_axis) * 1.02])

# Set tick locators
ax.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
ax.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
ax.xaxis.set_major_locator(MultipleLocator(10000))  # Set major ticks every 10000 units
ax.xaxis.set_minor_locator(MultipleLocator(1000))   # Set minor ticks every 1000 units

# Set tick parameters for ax
ax.tick_params(axis='both', which='minor', direction='in', width=0.75, length=4, bottom=False, top=True, left=True, right=True)
ax.tick_params(axis='x', which='minor', direction='in', width=0.75, length=2, bottom=True)  
ax.tick_params(axis='both', which='major', direction='in', width=1, length=9, bottom=False, top = True, left=True, right=True, labelsize=13)
ax.tick_params(axis='x', which='major', direction='in', width=1, length=4.5, bottom=True, labelsize=15)  

# Add a legend
ax.legend(fontsize=14)

# # Add labels and title
# ax.set_xlabel('Monte Carlo Threshold',fontsize = 15)
ax.set_ylabel('No of YSOs',fontsize = 20)
ax.set_title(f'Cluster {cluster_no}\nVariation of true members and false positive detections',fontsize = 22)

# Set grid lines thinner and fainter
ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

ax_res.plot(x_axis, residual, color = 'black', linestyle = '-.', label='Residual')
ax_res.get_xticklabels()

ax_res.set_xlabel('Monte Carlo Threshold', fontsize=20)
ax_res.set_ylabel('Residual YSOs\n[6] - [5]', fontsize=18)

# Calculate the range of the data
y_range_res = max(residual) - min(residual)

# Determine the appropriate tick interval based on the range
if 20 <= y_range_res <= 40:
    major_tick_interval = 5
    minor_tick_interval = 2.5
elif 40 < y_range_res <= 60:
    major_tick_interval = 12
    minor_tick_interval = 3
elif 60 < y_range_res <= 300:
    major_tick_interval = 20
    minor_tick_interval = 5
elif 10 <= y_range_res < 20:
    major_tick_interval = 4
    minor_tick_interval = 1
else:
    major_tick_interval = 1
    minor_tick_interval = 0.5

# Set tick locators
ax_res.yaxis.set_major_locator(MultipleLocator(major_tick_interval))
ax_res.yaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
ax_res.xaxis.set_major_locator(MultipleLocator(10000))  # Set major ticks every 10000 units
ax_res.xaxis.set_minor_locator(MultipleLocator(1000))   # Set minor ticks every 1000 units

# Set tick parameters for ax_res
ax_res.tick_params(axis='both', which='minor', direction='in', width=0.75, length=4, bottom=True, top=False, left=True, right=True)
ax_res.tick_params(axis='x', which='minor', direction='in', width=0.75, length=2, top=True)  
ax_res.tick_params(axis='both', which='major', direction='in', width=1, length=9, bottom=True, top=False, left=True, right=True, labelsize=13)
ax_res.tick_params(axis='x', which='major', direction='in', width=1, length=4.5, top=True, labelsize=15)  

ax_res.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)

# Adjust layout
plt.tight_layout()

plot_filename_mspec = f"mspec_{cluster_name}.svg"
plt.savefig(plot_filename_mspec, format = 'svg', dpi=300, bbox_inches='tight')

plot_filename_mspec = f"mspec_{cluster_name}.png"
plt.savefig(plot_filename_mspec, format = 'png', dpi=300, bbox_inches='tight')

plt.close()
gc.collect()

# ========================================Printing Results Here==================================================

print("\n")
print("===========Spatial Optimization Result====================")
print("Best MS fit parameters:", ms_fit_params)
print(" All best fit params are as follows: ",  best_fit_params)

print("===========Proper motion optimization Result====================")
print("Best MS fit parameters:", ms_fit_params_pm)
print(" All best fit params are as follows: ",  best_fit_params_pm)
print("\n")

#====================================================== Marking real cluster members===============================================
print("Marking observed cluster members...")
os.chdir(cluster_dir)

real_df = real.to_pandas()

membership_record_100k = np.load(f"membership_record_100k_{cluster_name}.npy", allow_pickle=True).item()
print("100k dictionary loaded successfully.")

center_dist_good = np.array([dist for dist, m in zip(best_fit_params[0:top_num,0], mask.flatten()) if m])
center_dist_limits = [min(center_dist_good),max(center_dist_good)]

no_of_iterations = 100000
storage_100k = {}

for i in range(len(membership_record_100k)):

    # Store data in the dictionary
    storage_100k[f'iter_{i}'] = real_df[membership_record_100k[i] > MC_thr]

# # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
# # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

# Get the keys from storage corresponding to iter_1 to iter_10
keys = ['iter_' + str(i) for i in range(0, runs)]

# Initialize common_members with the 'source_id' column from the first iteration
common_members_100k = storage_100k[keys[0]][['source_id']]

# Iterate over the keys and merge each 'source_id' column with common_members
for key in keys[1:]:
    common_members_100k = pd.merge(common_members_100k, storage_100k[key][['source_id']], on='source_id', how='inner')

# Identify outliers in the cluster data
ind = real_df['source_id'].isin(common_members_100k['source_id'])
cluster_data = real_df[ind]
outliers_cluster_sim = outlier_detect_observed(center_dist_limits, cluster_radius_deg, real, dist_limits , PMRA_limits, PMDEC_limits).to_pandas()
outer = cluster_data['source_id'].isin(outliers_cluster_sim['source_id'])

# Initialize mem_flag with zeros
mem_flag = np.zeros(len(real), dtype=int)

# Mark members with 1 (non-outliers)
member_indices = real_df['source_id'].isin(cluster_data['source_id'][~outer])
mem_flag[member_indices] = 1

# Add the mem_flag column to the DataFrame
real_df['mem_flag'] = mem_flag

# Convert back to an astropy table
real = Table.from_pandas(real_df)

# Print the number of detected members
print(f"Total detected members are: {np.sum(mem_flag)}")
print(f'Member stars declared as field stars: {len(cluster_data[outer])}')

# Write the updated table to a FITS file
real.write(f'updated_{cluster_name}_region.fits', overwrite=True)