import os
import numpy as np
import astropy.units as u
import astropy.coordinates as coord
import configparser
from astropy.coordinates import galactocentric_frame_defaults, SkyCoord, SphericalRepresentation
from astropy.table import Table, Column, vstack
from astropy.io import fits
from astropy.modeling.models import KingProjectedAnalytic1D
from scipy.stats import gaussian_kde, invweibull, median_abs_deviation
from scipy.optimize import linear_sum_assignment

# Set the galactocentric frame defaults
galactocentric_frame_defaults.set('latest')

config = configparser.ConfigParser()
config_path = "/pool/sao/vpatel/Cluster_Archive/config.ini"  # Give full path of config.ini file

if not os.path.exists(config_path):
    print(f"Error: Config file not found at {config_path} in cluster_model.py")
else:
    config.read(config_path)

# Set the random seed for reproducibility
np.random.seed(42)

# Define the directory path
directory = config["paths"]["sim_data_dir"]

# Check if the directory exists
if not os.path.exists(directory):
    # If the directory doesn't exist, create it
    os.makedirs(directory)

# Change the current working directory to the new directory
os.chdir(directory)

# Read the data file
tau_aur_gaia = Table.read('more_real_pm_cut_2.fits')


# ====================================================Helper functions====================================

def assignment_optimize(cost_matrix):
    '''
    Input: simulation - observed cost matrix
    Output: The row index is tp be applied on the observed data
    '''
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sorted_indices = np.argsort(col_ind)
    # Use these indices to sort both arrays
    # sorted_col_index = col_ind[sorted_indices]
    sorted_row_ind = row_ind[sorted_indices]
    return sorted_row_ind

def syn(delta):
    mad_delta = median_abs_deviation(delta,axis = None)
    if mad_delta == 0:
        mad_delta = np.finfo(float).eps  # Avoid division by zero
    syn_dist_square = (delta/mad_delta)**2
    return syn_dist_square

import numpy as np

def error_assign(dist_50, d16, d50, d84, *args):
    # Ensure arrays are numpy arrays
    dist_50 = np.array(dist_50)
    d16 = np.array(d16)
    d50 = np.array(d50)
    d84 = np.array(d84)
    
    if args:
        backup_16, backup_50 = map(np.array, args)
        
        # Calculate condition mask
        condition_mask = (d50 - d16) <= dist_50
        
        # Calculate dist_16 using np.where to apply condition mask
        dist_16 = np.where(condition_mask, dist_50 - (d50 - d16), dist_50 - (backup_50 - backup_16))
    else:
        dist_16 = dist_50 - (d50 - d16)
    
    dist_84 = dist_50 + (d84 - d50)

    return dist_50, dist_16, dist_84

def outlier_detect(self,cluster_real, dist_limits , PMRA_limits, PMDEC_limits):

    cluster_upper = self.center_distance + self.r_tide.value
    cluster_lower = self.center_distance - self.r_tide.value

    mask_nonmem = (
            (cluster_upper < dist_limits[0]) | 
            (cluster_lower > dist_limits[1]) | 
            (cluster_real['pmra'] < PMRA_limits[0]) | 
            (cluster_real['pmra'] > PMRA_limits[1]) | 
            (cluster_real['pmdec'] < PMDEC_limits[0]) | 
            (cluster_real['pmdec'] > PMDEC_limits[1])
        )
    outliers_cluster_real = cluster_real[mask_nonmem]

    return outliers_cluster_real

def fit_kde_and_sample(kde_pmra, kde_pmdec, kde_parallax, n_samples):
    """
    Fit a KDE to the data and draw samples from it.

    Parameters:
    pmra (array-like): Proper motion in RA to fit the KDE to.
    pmdec (array-like): Proper motion in DEC to fit the KDE to.
    parallax (array-like): Parallax data to fit the KDE to.
    n_samples (int): The number of samples to draw from the fitted KDE.
    directory (str): Directory where the data is located.
    bandwidth (float, optional): The bandwidth of the KDE. If None, the bandwidth is automatically determined.

    Returns:
    extra_pm (Table): A table containing the sampled data for pmra, pmdec, and parallax.
    """

    # # Fit the KDE to the data
    # kde_pmra = gaussian_kde(pmra, bw_method=bandwidth)
    # kde_pmdec = gaussian_kde(pmdec, bw_method=bandwidth)
    # kde_parallax = gaussian_kde(parallax, bw_method=bandwidth)

    # Set the random seed if provided
    np.random.seed(50)

    # Draw samples from the fitted KDE
    samples_pmra = kde_pmra.resample(n_samples).T
    samples_pmdec = kde_pmdec.resample(n_samples).T
    samples_parallax = kde_parallax.resample(n_samples).T

    extra_pm = Table()
    extra_pm['pmra'] = Column(data = samples_pmra)
    extra_pm['pmdec'] = Column(data = samples_pmdec)
    extra_pm['parallax'] = Column(data = samples_parallax)
    
    return extra_pm


def newpm_generate_multiple(self, kde_pmra, kde_pmdec, kde_parallax):
    os.chdir(directory+ f"/{self.cluster_name}_sim")
    
    if self.num_stars > len(tau_aur_gaia['pmra']):
        n = self.num_stars - len(tau_aur_gaia['pmra'])

        # extra_pm = fit_kde_and_sample(tau_aur_gaia['pmra'], tau_aur_gaia['pmdec'], tau_aur_gaia['parallax'], n)
        extra_pm = fit_kde_and_sample(kde_pmra, kde_pmdec, kde_parallax, n)

        # Convert extra_pm columns to match tau_aur_gaia column shapes and data types
        extra_pm['pmra'] = np.array(extra_pm['pmra']).astype('>f8').flatten()
        extra_pm['pmdec'] = np.array(extra_pm['pmdec']).astype('>f8').flatten()
        extra_pm['parallax'] = np.array(extra_pm['parallax']).astype('>f8').flatten() 

        newpm_table = vstack([tau_aur_gaia['pmra', 'pmdec', 'parallax'], extra_pm])
        
        del extra_pm

    else:
        n = self.num_stars
        np.random.seed(6)
        newpm_table = tau_aur_gaia['pmra', 'pmdec', 'parallax'][np.random.choice(len(tau_aur_gaia), size=n, replace=False)]
    
    return newpm_table

#======================================================================Cluster Models============================================

## King2D model- Isothermel Sphere 
def King2D(self, kde_pmra, kde_pmdec, kde_parallax, Wpermit):

    amplitude = 1

    # Initialize the KingProjectedAnalytic1D model
    mod = KingProjectedAnalytic1D(amplitude=amplitude, r_core=self.r_core.value, r_tide=self.r_tide.value)

    # Generate a set of radii
    n_points = 10000
    r_values = np.linspace(0.001, self.r_tide.value, n_points)

    # Calculate the King profile values for the given radii
    king_values = mod(r_values)

    # Use Gaussian KDE to approximate the King profile
    kde = gaussian_kde(r_values, weights=king_values, bw_method=0.00001)

    # # Sample distances from the KDE
    distances = kde.resample(self.num_stars).flatten() * u.pc

    theta = np.random.uniform(0, 360, self.num_stars) * u.deg
    phi = (np.arccos(2 * np.random.uniform(0, 1, self.num_stars) - 1) - (np.pi/2)) * 180/np.pi * u.deg 

    center = SphericalRepresentation(self.center_ra, self.center_dec, self.center_distance*u.pc)
    offset = SphericalRepresentation(theta, phi , distances)

    offset_coord = SkyCoord(center + offset, frame = 'icrs')

    # Extracting RA, Dec, and distance values for plotting
    ras = offset_coord.ra.deg
    decs = offset_coord.dec.deg
    distances = offset_coord.distance.pc

    # ===============================================================Adding proper motion information====================================

    newpm_table = newpm_generate_multiple(self, kde_pmra, kde_pmdec, kde_parallax)


    newpm_table['sim_pmra'] = self.pm_scale * (newpm_table['pmra'] * 1000)/(newpm_table['parallax'] * distances)
    newpm_table['sim_pmdec'] = self.pm_scale * (newpm_table['pmdec'] * 1000)/(newpm_table['parallax'] * distances)


    c = coord.SkyCoord(ra = ras * u.degree,
                dec = decs * u.degree,
                distance = distances * u.pc,
                frame='icrs')
    gal = c.transform_to(coord.Galactocentric(galcen_distance=8.178*u.kpc)) 
    galactic_coord = c.galactic

    temp_data = {
        'RA': ras,
        'Dec': decs,
        'Distance': distances,
        'PMRA': np.array(newpm_table['sim_pmra']),
        'pmra_error': np.full(len(ras), np.nan),  # Empty column filled with NaNs
        'PMDEC': np.array(newpm_table['sim_pmdec']),
        'pmdec_error': np.full(len(ras), np.nan),
        'X': gal.x.value,
        'Y': gal.y.value,
        'Z': gal.z.value,
        'mem_flag': np.ones(len(ras), dtype=int),
        'L': galactic_coord.l.deg,
        'B':galactic_coord.b.deg
    }

    cluster_model = Table(temp_data)

    if Wpermit ==1:
        cluster_model.write(f'simulated_cluster_{self.cluster_name}.fits', format='fits', overwrite=True)

    return cluster_model


##================================================================ Code for generating field YSOs ================================================


def generate_noise(self, cluster_model, Wpermit):
    
    cluster_real = Table.read(directory + f'/{self.cluster_name}_sim/{self.cluster_name}_region_cleaned.fits')

    center_coords = SkyCoord(ra = self.center_ra, dec = self.center_dec, frame = 'icrs')


    ##############################################################################################

    cluster_coords = SkyCoord(cluster_real['ra_epoch2000_x']*u.deg,cluster_real['dec_epoch2000_x']*u.deg)


    # Calculate distances of each point from the cluster center
    distances = center_coords.separation(cluster_coords)

    # Find the index of the point farthest from the center
    farthest_point_index = np.argmax(distances)
    farthest_point_distance = distances[farthest_point_index]

    # Determine the cluster radius
    search_radius = farthest_point_distance.to(u.deg).value 

    ###############################################################################################
    # Generate random distances and angles
    r = np.sqrt(np.random.uniform(0, 1, 2 * self.noise_ysos)) * search_radius
    theta = np.random.uniform(0, 360, 2 * self.noise_ysos) * u.deg

    # Convert polar coordinates to Cartesian coordinates on the unit sphere
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Calculate the offset in RA and Dec
    delta_ra = (x / np.cos(self.center_dec)) 
    delta_dec = y 

    # Calculate the new RA and Dec
    ras_N = self.center_ra + delta_ra * u.deg
    decs_N = self.center_dec + delta_dec * u.deg

    # Create SkyCoord object for the new coordinates
    offset_coord = SkyCoord(ra=ras_N, dec=decs_N, frame='icrs')

    # Wrap RA coordinates at 360 degrees
    ras_N_wrapped = offset_coord.ra.wrap_at(360 * u.deg)
    decs_N = offset_coord.dec

    # Extracting RA and Dec values for plotting
    ras_N = ras_N_wrapped.deg
    decs_N = decs_N.deg

    # Generate random noise distances within the specified range
    # Modification: Date June 22: Removal of dependence on self.center_distance of noise upper and lower ranges
    # Given below is the previous version:
    # min_distance = self.noise_lower * self.center_distance * u.pc
    # max_distance = self.noise_upper * self.center_distance * u.pc

    min_distance = self.noise_lower * u.pc
    max_distance = self.noise_upper * u.pc
    
    noise_distances = np.random.uniform(min_distance.value, max_distance.value, 2*self.noise_ysos)

    ## The following part of the function creates cavity of noise points in the region where cluster stars 
    ## are present to prevent any confusion while analyzing the simulation results.

    # Create a SkyCoord object for the center of the spherical cluster
    c = SkyCoord(ra=self.center_ra, dec=self.center_dec, distance=self.center_distance*u.pc, frame='icrs')

    # Create a catalog of SkyCoord objects for the generated points
    catalog = SkyCoord(ra=ras_N * u.degree, dec=decs_N * u.degree, distance = noise_distances * u.pc)

    # Calculate the separation between each coordinate in the catalog and the center coordinate
    separations = c.separation_3d(catalog)

    # Create a mask to filter out points outside the cluster radius
    mask = separations > self.r_tide

    # Apply the mask to consider only points outside the cluster radius
    ras_N = np.random.choice(ras_N[mask],size = self.noise_ysos)
    decs_N = np.random.choice(decs_N[mask],size = self.noise_ysos)
    noise_distances = np.random.choice(noise_distances[mask],size = self.noise_ysos)

    c = coord.SkyCoord(ra = ras_N * u.degree,
                dec = decs_N * u.degree,
                distance = noise_distances * u.pc,
                frame='icrs')
    
    gal = c.transform_to(coord.Galactocentric(galcen_distance=8.178*u.kpc)) 
    galactic_coord = c.galactic # This line of code is for galactic l and b coordinates of ideal noise


    cluster_noise = Table({
        'RA': ras_N,
        'Dec': decs_N,
        'Distance': noise_distances,
        'PMRA': np.full(len(ras_N), np.nan),
        'pmra_error': np.full(len(ras_N), np.nan),
        'PMDEC': np.full(len(ras_N), np.nan),
        'pmdec_error': np.full(len(ras_N), np.nan),
        'X': gal.x.value,
        'Y': gal.y.value,
        'Z': gal.z.value,
        'mem_flag': np.zeros(len(ras_N), dtype=int),
        'L': galactic_coord.l.deg,
        'B':galactic_coord.b.deg
    })

    if Wpermit == 1:

        cluster_noise = Table({
        'RA': ras_N,
        'Dec': decs_N,
        'Distance': noise_distances,
        'PMRA':np.empty(len(ras_N), dtype=int),
        'pmra_error': np.empty(len(ras_N), dtype=int),
        'PMDEC': np.empty(len(ras_N), dtype=int),
        'pmdec_error': np.empty(len(ras_N), dtype=int),
        'X': gal.x.value,
        'Y': gal.y.value,
        'Z': gal.z.value,
        'mem_flag': np.zeros(len(ras_N), dtype=int),
        'L': galactic_coord.l.deg,
        'B':galactic_coord.b.deg
    })

        # # Save the FITS table to a file
        cluster_noise.write(f'simulated_noise_{self.cluster_name}.fits', overwrite=True)

    region_simulation = vstack([cluster_model,cluster_noise])

    # Create SkyCoord objects for real and simulated data points
    coord_real = SkyCoord(l=cluster_real['l_epoch2000'] * u.deg, b=cluster_real['b_epoch2000'] * u.deg, frame='galactic')
    coord_sim = SkyCoord(l=region_simulation['L'] * u.deg, b=region_simulation['B'] * u.deg, frame='galactic')

    # Calculate the separation matrix in a vectorized way
    separation_matrix = coord_sim.separation(coord_real[:, np.newaxis]).deg
    del coord_real, coord_sim

    sorted_row_ind = assignment_optimize(separation_matrix)
    del separation_matrix 


    # Fill missing values with averages using np.where in single lines
    G_sim = np.where(np.isnan(cluster_real['phot_g_mean_mag'][sorted_row_ind]), np.nanmean(cluster_real['phot_g_mean_mag']), cluster_real['phot_g_mean_mag'][sorted_row_ind])
    G_err_sim = np.where(np.isnan(sorted_row_ind), np.nanmean(cluster_real['e_Gmag']), cluster_real['e_Gmag'][sorted_row_ind])
    bp_rp_sim = np.where(np.isnan(cluster_real['bp_rp'][sorted_row_ind]), np.nanmean(cluster_real['bp_rp']), cluster_real['bp_rp'][sorted_row_ind])
    bp_rp_err_sim = np.where(np.isnan(cluster_real['e_BPmag'][sorted_row_ind]) | np.isnan(cluster_real['e_RPmag'][sorted_row_ind]), np.nanmean(np.abs(cluster_real['e_BPmag']) + np.abs(cluster_real['e_RPmag'])), np.abs(cluster_real['e_BPmag'][sorted_row_ind]) + np.abs(cluster_real['e_RPmag'][sorted_row_ind]))

    # Add columns to the region_simulation table
    region_simulation.add_column(Column(G_sim, name='G'))
    region_simulation.add_column(Column(G_err_sim, name='G_err'))
    region_simulation.add_column(Column(bp_rp_sim, name='bp_rp'))
    region_simulation.add_column(Column(bp_rp_err_sim, name='bp_rp_err'))

    if Wpermit ==1:
        region_simulation['X','Y','Z','RA', 'Dec','Distance', 'PMRA','pmra_error','PMDEC','pmdec_error', 'mem_flag','L','B', 'G','G_err','bp_rp', 'bp_rp_err'].write(f'region_simulation_{self.cluster_name}.fits', format = 'fits' , overwrite = 'True')
    
    return region_simulation
    
## ===================================================== Code for error extraction ===========================================

def real_error(self, region_simulation, real_data, Wpermit, update_permit, dist_limits , PMRA_limits, PMDEC_limits, mean_PMRA, mean_PMDEC):

    cluster_real = Table.read(directory + f'/{self.cluster_name}_sim/{self.cluster_name}_region_cleaned.fits')

    sim_data = region_simulation

    dists = sim_data['Distance']
    distr = real_data['final_dist']
    delta_d = np.abs(dists - distr[:,np.newaxis])

    Gs = sim_data['G']
    Gr = np.where(np.isnan(real_data['phot_g_mean_mag']), np.nanmean(real_data['phot_g_mean_mag']), real_data['phot_g_mean_mag'])
    delta_G = np.abs(Gs - Gr[:,np.newaxis])

    bp_rps = sim_data['bp_rp']
    bp_rpr = np.where(np.isnan(real_data['bp_rp']), np.nanmean(real_data['bp_rp']), real_data['bp_rp'])
    delta_bp_rp = np.abs(bp_rps - bp_rpr[:,np.newaxis])

    synthetic_dist_matrix= np.sqrt(syn(delta_d) + syn(delta_G) + syn(delta_bp_rp))

    sorted_row_ind = assignment_optimize(synthetic_dist_matrix)
    
    dist_hi = real_data['final_dist_hi'][sorted_row_ind]
    dist_lo = real_data['final_dist_lo'][sorted_row_ind]
    closest_dist = real_data['final_dist'][sorted_row_ind]

    ## This part of the code is to calculate distance of real full data similar to simulated distance
    # Only distances are considered for calculating cost matrix 
    ###########################################################
    synthetic_dist_matrix_backup = delta_d
    sorted_row_ind_x = assignment_optimize(synthetic_dist_matrix_backup)
    
    backup_lo = real_data['final_dist_lo'][sorted_row_ind_x]
    backup_med = real_data['final_dist'][sorted_row_ind_x]
    ###########################################################

    d_50, d_16, d_84 = error_assign(sim_data['Distance'], dist_lo, closest_dist, dist_hi, backup_lo, backup_med)

    # Append the lists as new columns to the simulated data
    sim_data['closest_G'] = Gr[sorted_row_ind]
    sim_data['closest_bp_rp'] = bp_rpr[sorted_row_ind]
    sim_data['closest_median_dist'] = closest_dist
    sim_data['median_dist_lo'] = d_16
    sim_data['median_dist_hi'] = d_84

    # Parameters for inverse Weibul distribution
    # Estimate shape parameter (k) using the 16th percentile
    k = np.log(np.log(0.84) / np.log(0.16)) / np.log(d_16 / d_84)

    # Estimate scale parameter (λ) using the 50th percentile
    λ = d_50 * np.log(2)**(1/k) 

    # Generate random samples from the Weibull distribution

    # Initialize an empty list to store perturbed distances for each pair of k and λ
    perturbed_cluster_dists = []


    # Iterate over each pair of k and λ
    for count,(k_val, λ_val) in enumerate(zip(k, λ)):
        # Generate perturbed cluster distances for the current pair

        np.random.seed(6)
        perturbed_cluster_dist = invweibull.rvs(c=k_val, scale=λ_val, size=10**5)
        rem = count % len(np.array([80,68,30,15,5]))
        # np.arange(85,4,-5)
        # 85,77,70,67,60,35,27,20,15,5
        # 80,68,30,15,5
        # 80,70,60,30,20,12,5
        percentile = np.array([80,68,30,15,5])
        lower_percentile = percentile[rem] - 3
        mask = (perturbed_cluster_dist < np.percentile(perturbed_cluster_dist, percentile[rem])) & \
        (perturbed_cluster_dist > np.percentile(perturbed_cluster_dist, lower_percentile))
         
        np.random.seed(6)

        sample = np.random.choice(perturbed_cluster_dist[mask])
        perturbed_cluster_dists.append(sample)

    del rem

    perturbed_cluster_dists = np.array(perturbed_cluster_dists)
    sim_data['perturbed_dist'] = perturbed_cluster_dists

    if Wpermit ==1:
        sim_data.write(f'sim_data_median_{self.cluster_name}.fits', format='fits', overwrite=True)  

    ## Calculating final distance errors for simulated_cluster

    # # Initialize an empty separation matrix (2D array)
    num_real = len(cluster_real)
    num_sim = len(sim_data)
    synthetic_dist_matrix = np.zeros((num_real, num_sim))

    ls = sim_data['L']
    lr = cluster_real['l_epoch2000']
    delta_l = np.abs(ls - lr[:,np.newaxis])

    bs = sim_data['B']
    br = cluster_real['b_epoch2000']
    delta_b = np.abs(bs - br[:,np.newaxis])

    dists = sim_data['perturbed_dist']
    distr = cluster_real['final_dist']
    delta_d = np.abs(dists - distr[:,np.newaxis])

    # Gr = np.where(np.isnan(cluster_real['phot_g_mean_mag']), np.nanmean(cluster_real['phot_g_mean_mag']), cluster_real['phot_g_mean_mag'])
    # delta_G = np.abs(Gs - Gr[:,np.newaxis])

    # bp_rpr = np.where(np.isnan(cluster_real['bp_rp']), np.nanmean(cluster_real['bp_rp']), cluster_real['bp_rp'])
    # delta_bp_rp = np.abs(bp_rps - bp_rpr[:,np.newaxis])

    # synthetic_dist_matrix= np.sqrt(syn(delta_l) + syn(delta_b) + syn(delta_d) + syn(delta_G) + syn(delta_bp_rp))
    synthetic_dist_matrix= np.sqrt(syn(delta_l) + syn(delta_b) + syn(delta_d)) 

    sorted_row_ind = assignment_optimize(synthetic_dist_matrix)

    final_dist_lo = cluster_real['final_dist_lo'][sorted_row_ind]
    final_dist_hi = cluster_real['final_dist_hi'][sorted_row_ind]
    final_closest_dist = cluster_real['final_dist'][sorted_row_ind]

    # PMRA_error = cluster_real['pmra_error'][sorted_row_ind]
    # closest_PMRA = cluster_real['pmra'][sorted_row_ind]
    # PMDEC_error = cluster_real['pmdec_error'][sorted_row_ind]
    # closest_PMDEC = cluster_real['pmdec'][sorted_row_ind]

    ## This part of the code is to calculate distance of real full data similar to simulated distance
    # Only distances are considered for calculating cost matrix 
    ###########################################################
    dists = sim_data['perturbed_dist']
    distr = real_data['final_dist']
    delta_D = np.abs(dists - distr[:,np.newaxis])
    synthetic_dist_matrix_backup = delta_D
    sorted_row_ind_xxx  = assignment_optimize(synthetic_dist_matrix_backup)
    
    backup_lo = real_data['final_dist_lo'][sorted_row_ind_xxx]
    backup_med = real_data['final_dist'][sorted_row_ind_xxx]
    ###########################################################

    final_d_50, final_d_16, final_d_84 = error_assign(sim_data['perturbed_dist'], final_dist_lo, final_closest_dist, final_dist_hi, backup_lo, backup_med)

    # Append the lists as new columns to the simulated data
    sim_data['final_perturbed_closest_dist'] = final_closest_dist
    sim_data['perturbed_dist'] = final_d_50
    sim_data['final_dist_lo'] = final_d_16
    sim_data['final_dist_hi'] = final_d_84
       
    # # Append the lists as new columns to the simulated data
    # sim_data['closest_PMRA'] = closest_PMRA
    # # sim_data['PMRA'] = closest_PMRA
    # sim_data['PMRA_error'] = PMRA_error

    # sim_data['closest_PMDEC'] = closest_PMDEC
    # # sim_data['PMDEC'] = closest_PMDEC
    # sim_data['PMDEC_error'] = PMDEC_error

    cluster_id = sim_data['mem_flag']==1
    outliers_cluster_real = outlier_detect(self, cluster_real, dist_limits , PMRA_limits, PMDEC_limits)

    # Create a boolean mask where the source_id in cluster_real is NOT in outliers_cluster_real
    mask = np.isin(cluster_real['source_id'], outliers_cluster_real['source_id'])

    # Filter the cluster_real table to remove rows where source_id matches those in outliers
    filtered_cluster_real = cluster_real[~mask]

    sim_data['PMRA'][cluster_id] = sim_data['PMRA'][cluster_id] - np.median(sim_data['PMRA'][cluster_id]) + mean_PMRA
    sim_data['PMDEC'][cluster_id] = sim_data['PMDEC'][cluster_id] - np.median(sim_data['PMDEC'][cluster_id]) + mean_PMDEC


    # coord_real_pm = SkyCoord(ra=filtered_cluster_real['pmra'].astype(np.float32) * u.deg, 
    #                         dec=filtered_cluster_real['pmdec'].astype(np.float32) * u.deg, frame='icrs')
    # coord_sim_pm = SkyCoord(ra=sim_data['PMRA'][cluster_id].astype(np.float32) * u.deg, 
    #                         dec=sim_data['PMDEC'][cluster_id].astype(np.float32) * u.deg, frame='icrs')

    # separation_matrix_pm = coord_sim_pm.separation(coord_real_pm[:, np.newaxis]).mas
    
    pmra_s = sim_data['PMRA'][cluster_id].astype(np.float32) 
    pmra_r = filtered_cluster_real['pmra'].astype(np.float32)
    delta_pmra = (pmra_s - pmra_r[:,np.newaxis])**2

    pmdec_s = sim_data['PMDEC'][cluster_id].astype(np.float32) 
    pmdec_r = filtered_cluster_real['pmdec'].astype(np.float32)
    delta_pmdec = (pmdec_s - pmdec_r[:,np.newaxis])**2

    separation_matrix_pm = np.sqrt(delta_pmra + delta_pmdec)

    # Free SkyCoord objects to save memory
    del delta_pmra, delta_pmdec, pmra_s, pmra_r, pmdec_s, pmdec_r

    sorted_row_ind_pm = assignment_optimize(separation_matrix_pm)
    del separation_matrix_pm

    sim_data['pmra_error'][cluster_id] = filtered_cluster_real['pmra_error'][sorted_row_ind_pm]
    sim_data['pmdec_error'][cluster_id] = filtered_cluster_real['pmdec_error'][sorted_row_ind_pm]

    # Now error for simululated cluster members is assigned
    matched_indices_pm = np.isin(cluster_real['source_id'], filtered_cluster_real['source_id'][sorted_row_ind_pm]) 

    cluster_real_F = cluster_real[~matched_indices_pm]
    distr_field = cluster_real_F['final_dist']
    dists_field = sim_data['perturbed_dist'][~cluster_id]

    delta_d_field = np.abs(dists_field - distr_field[:,np.newaxis])

    sorted_row_ind_pm_F = assignment_optimize(delta_d_field)

    # Extract PMRA and PMDEC for non-matched indices
    sim_data['PMRA'][~cluster_id] = cluster_real_F['pmra'][sorted_row_ind_pm_F]
    sim_data['pmra_error'][~cluster_id] = cluster_real_F['pmra_error'][sorted_row_ind_pm_F]
    sim_data['PMDEC'][~cluster_id] = cluster_real_F['pmdec'][sorted_row_ind_pm_F]
    sim_data['pmdec_error'][~cluster_id] = cluster_real_F['pmdec_error'][sorted_row_ind_pm_F]
    
    c = coord.SkyCoord(ra = np.array(sim_data['RA']) * u.degree,
                dec = sim_data['Dec'] * u.degree,
                distance = sim_data['perturbed_dist'] * u.pc,
                frame='icrs')
    gal = c.transform_to(coord.Galactocentric(galcen_distance=8.178*u.kpc)) 

    X = gal.x
    Y = gal.y
    Z = gal.z
    sim_data['X'] = X
    sim_data['Y'] = Y
    sim_data['Z'] = Z

    if Wpermit == 1:
        sim_data.write(f'sim_full_{self.cluster_name}.fits', format='fits', overwrite=True)  

    sim_data.rename_column('RA','ra_epoch2000_x')
    sim_data.rename_column('Dec','dec_epoch2000_x')
    sim_data.rename_column('perturbed_dist','final_dist')
    sim_data.rename_column('PMRA','pmra')
    sim_data.rename_column('PMDEC','pmdec')
    sim_data.rename_column('L','l_epoch2000')
    sim_data.rename_column('B','b_epoch2000')


    # Create a new column 'source_id' with unique identifiers
    sim_data['source_id'] = [int(str(row_num) + str(mem_flag)) for row_num, mem_flag in enumerate(sim_data['mem_flag'])]
    sim_req = sim_data[['source_id','ra_epoch2000_x','dec_epoch2000_x','final_dist','final_dist_lo','final_dist_hi','pmra','pmra_error','pmdec','pmdec_error','X','Y','Z','mem_flag','l_epoch2000','b_epoch2000', 'G','G_err','bp_rp', 'bp_rp_err']]

    if update_permit == 1:
        # outliers_cluster_real = outlier_detect(self, cluster_real, dist_limits , PMRA_limits, PMDEC_limits)

        distsim = sim_req[sim_req['mem_flag'] == 0]['final_dist']
        distreal_outliers = outliers_cluster_real['final_dist']
        delta_Dist = np.abs(distreal_outliers - distsim[:, np.newaxis])

        row_ind_xx, col_ind_xx = linear_sum_assignment(delta_Dist)
        sorted_indices_xx = np.argsort(col_ind_xx)
        sorted_row_ind_xx = row_ind_xx[sorted_indices_xx]

        # Specify the columns to update
        columns_to_update = [
            'ra_epoch2000_x', 'dec_epoch2000_x', 'final_dist', 'final_dist_lo', 'final_dist_hi',
            'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'X', 'Y', 'Z', 'l_epoch2000', 'b_epoch2000'
        ]
        # 
        # Find the indices where 'mem_flag' is 0 in sim_req
        mem_flag_indices = sim_req['mem_flag'] == 0

        # Sort rows of sim_req based on sorted_row_ind_xx (make sure sorted_row_ind_xx is an array of indices)
        sorted_indices = np.where(mem_flag_indices)[0][sorted_row_ind_xx]

        # Update the columns in sim_req using outliers_cluster_real values
        for column in columns_to_update:
            sim_req[column][sorted_indices] = outliers_cluster_real[column]

    if Wpermit == 1:
        sim_req.write(f'required_sim_data_{self.cluster_name}.fits', format='fits', overwrite=True)  

    return sim_req