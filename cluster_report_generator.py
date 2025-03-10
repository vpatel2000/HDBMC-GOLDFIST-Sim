# Standard Library Imports
import os
import shutil
import subprocess
import warnings
import importlib

# Third-Party Library Imports
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FuncFormatter
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyUserWarning

# Suppress specific warnings
warnings.filterwarnings('ignore', category=AstropyUserWarning)

# Custom Module Imports
import HDBSCAN_func  # Import the module initially
import cluster_model

# Reload Custom Modules (if modifications occur dynamically)
importlib.reload(HDBSCAN_func)
from HDBSCAN_func import *

importlib.reload(cluster_model)
from cluster_model import config_path, config

def generate_pdf(tex_file):
    """Generates a PDF file from a LaTeX file.

    Args:
    tex_file: The path to the LaTeX file.

    Returns:
    The path to the generated PDF file.
    """

    pdf_file = tex_file.replace('.tex', '.pdf')
    subprocess.run(['pdflatex', tex_file])
    return pdf_file


gcolor = ["orange","#18dbcb", "red", "#93f707","#18dbcb", "white", "#f00ea1", "goldenrod", "blue", "sienna", "#357362"] * 20
gmarker = ["o", "p", "s", "v", ">", "<", "p", "s", "d", 'o'] * 20

directory = config["paths"]["primary_dir"]

os.chdir(directory)

## Defining paths
data_path = "DATA/"


cluster_detail = Table.read(data_path+"YSO_cluster.fits").to_pandas()

# clusters = [257]
# radius_factor = [1.5]

clusters = [163]
radius_factor = [3]

# clusters = [123]
# radius_factor = [2.3]


for cluster,rf in zip(clusters,radius_factor):

    from astropy.table import Table,join


    sim_cluster_dir = directory+ f"/simulated_clusters/Cluster_{cluster}_sim"
    os.chdir(sim_cluster_dir)

    MC_table = pd.read_csv(f"MC_thr_Cluster_{cluster}.txt", delimiter='\t')

    MC_thr = MC_table['MC_thr'][0]
    # MC_thr = 13000
    MC_thr_1k = MC_thr/100

    os.chdir(directory)

    table=Table.read(data_path+"SFOG_plus_gaiaYSO_final_dist_XYZ(J2000)_good_astrometry_outer_galaxy_new.fits")
    ir_yso = Table.read(data_path+"IR_YSOs.fits")
    cluster_detail = Table.read(data_path+"YSO_cluster.fits").to_pandas()
    sfog_full_data = Table.read(data_path+"SFOG_catalog_to_use.fits")

    full_data = table.to_pandas()

    cluster_no = cluster
    cluster_name = f'Cluster_{cluster_no}'

    # cluster_name = cluster
    # cluster_no = cluster_name.split('_')[1]

    print(f"Analyzing Cluster {cluster_no}...")

    cluster_dir = directory+ f"/{cluster_name}"
    os.makedirs(cluster_dir, exist_ok=True)

    # Change the current working directory to the newly created directory
    os.chdir(f"{cluster_name}")

    #*********************************************************************
    ## Comment these lines when analyzing Cluster 123
    cen_RA = round(float(cluster_detail['Central RA'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0]),6)
    cen_Dec  = round(float(cluster_detail['Central Dec'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0]),6)
    #*********************************************************************

    #*********************************************************************
    ## Uncomment these center coordinates when analyzing Cluster 123
    # cen_RA = 312.957906
    # cen_Dec = 44.541232
    #*********************************************************************

    cluster_radius = float(cluster_detail['Circular Radius'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0])*rf

    # Create a SkyCoord object
    sky_coord = SkyCoord(ra=cen_RA*u.deg, dec=cen_Dec*u.deg, frame='icrs')

    # Convert to Galactic coordinates
    galactic_coord = sky_coord.transform_to('galactic')

    # Extract Galactic Longitude (l) and Latitude (b)
    cen_lon = galactic_coord.l.value
    cen_lat = galactic_coord.b.value

    sfog_cluster_data = sfog_full_data[sfog_full_data['Cluster']==int(cluster_no)]

    IR_no = round(cluster_detail['# YSO'][cluster_detail['Cluster #']== int(cluster_no)].iloc[0])

    # Convert central RA and Dec to SkyCoord object
    central_coord = SkyCoord(ra=cen_RA*u.deg, dec=cen_Dec*u.deg, frame='icrs')

    # Calculate angular separation between each point in ir_yso and central coordinates
    t_yso_coords = SkyCoord(ra=table['ra_epoch2000_x']*u.deg, dec=table['dec_epoch2000_x']*u.deg, frame='icrs')
    ir_yso_coords = SkyCoord(ra=ir_yso['RA_deg']*u.deg, dec=ir_yso['Dec_deg']*u.deg, frame='icrs')

    angular_separation_ir = central_coord.separation(ir_yso_coords)
    angular_separation_t = central_coord.separation(t_yso_coords)

    # Select points within the cluster radius
    ir_table = ir_yso[angular_separation_ir < cluster_radius*u.deg]
    t = table[angular_separation_t < cluster_radius*u.deg]

    # data = t[['ra','dec','final_dist','final_dist_lo','final_dist_hi','X','Y','Z','pmra','pmra_error','pmdec','pmdec_error']].to_pandas()
    # data = data.to_numpy()
    # data = t.to_pandas()

    data = Table.read(cluster_dir+ f'/updated_{cluster_name}_region.fits').to_pandas()
    #==================================================Creating Membership Records===================================
    # Step 1: Creating membership records.

    print("Generating membership records...........")
    print(f"{os.cpu_count()} cores have been put to work! So relax...")  

    if  os.path.isfile(f"membership_record_100k_{cluster_name}.npy"):
        # File exists, load the dictionary
        membership_record_100k = np.load(f"membership_record_100k_{cluster_name}.npy", allow_pickle=True).item()
        print("100k dictionary loaded successfully.")
    else:
        print("100k dictionary does not exist. Saving new dictionary.")
        # # Step 1: Creating membership records

        membership_record_100k = {}

        for i in range(10):
            membership_record_100k[i] = run_hdbscan_parallel(data, n=100000, cluster_no = 0, prob_thr = 0.1)
            print(f'Run: {i+1}. Completed 100000 iterations!')

        np.save(f'membership_record_100k_{cluster_name}.npy', membership_record_100k)


    if os.path.isfile(f"membership_record_1k_{cluster_name}.npy"):
        # File exists, load the dictionary
        membership_record_1k = np.load(f"membership_record_1k_{cluster_name}.npy", allow_pickle=True).item()
        print("1k dictionary loaded successfully.")
    else:
        print("1k dictionary does not exist. Saving new dictionary.")
        # # Step 1: Creating membership records

        membership_record_1k = {}

        for i in range(10):
            membership_record_1k[i] = run_hdbscan_parallel(data, n=1000, cluster_no = 0, prob_thr = 0.1)
            print(f'Run: {i+1}. Now completed 1000 iterations.')

        np.save(f'membership_record_1k_{cluster_name}.npy', membership_record_1k)

    #================================HDBSCAN-MC stability check and common member identification=================
    print("Performing HDBSCAN-MC stability check and common member identification...........")    



    annotation_coords = (0.02, 0.93)
    y = []
    no_of_bins = 100
    bw = 0.09
    threshold_percentage = 34
    no_of_iterations = 1000
    storage = {}

    for i in range(len(membership_record_1k)):
        
        hist, bin_edges = np.histogram(membership_record_1k[i], bins=no_of_bins, range=(0, no_of_iterations), density=True)
        
        # Create KDE
        kde = gaussian_kde(membership_record_1k[i], bw_method= bw)

        # Evaluate the KDE on a grid
        x_grid = np.linspace(0, no_of_iterations, 1000)
        y_kde = kde(x_grid)

        # Find minima using the scipy.signal.find_peaks function
        peaks, _ = find_peaks(y_kde)

        y.append(len(membership_record_1k[i][membership_record_1k[i] > MC_thr_1k]))
        
        # Store data in the dictionary
        storage[f'iter_{i}'] = data[membership_record_1k[i] > MC_thr_1k]


    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(18, 8))


    ax1.plot(range(1,11),y,linestyle ='-.',marker= 's',color = 'black' , markerfacecolor ='blue', markersize = 5)

    ax1.set_xlabel('No. of algorithm runs',fontsize = 22)
    ax1.set_ylabel('Member counts', fontsize = 22)

    # Add a subtitle below the main title
    subtitle = f'Iterations: 1000'
    ax1.annotate(subtitle, (0.5, 1.015), xycoords='axes fraction', fontsize=22, ha='center')

    ax1.yaxis.set_major_locator(MultipleLocator(20))
    ax1.yaxis.set_major_formatter('{x:.1f}')
    ax1.yaxis.set_minor_locator(MultipleLocator(10))

    ax1.tick_params(axis='both',which='minor',direction='in' ,width=0.75,length=4,bottom=True, top=True ,left=True, right=True, labelsize = 13)
    ax1.tick_params(axis='both',which='major',direction='in',  width=1,length=7,bottom=True, top=True ,left=True, right=True, labelsize = 15)

    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.xaxis.set_major_formatter('{x:.0f}')
    # ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax1.tick_params(axis='both',which='minor',direction='in' ,width=0.75,length=4,bottom=True, top=True ,left=True, right=True)
    ax1.tick_params(axis='both',which='major',direction='in',  width=1,length=7,bottom=True, top=True ,left=True, right=True)
    ax1.annotate('(a)', xy = annotation_coords, xycoords="axes fraction", fontsize=24, fontweight='normal')
    
    # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
    # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

    # Get the keys from storage corresponding to iter_1 to iter_10
    keys = ['iter_' + str(i) for i in range(0, 10)]

    # Initialize common_members with the 'source_id' column from the first iteration
    common_members_1k = storage[keys[0]][['source_id']]

    # Iterate over the keys and merge each 'source_id' column with common_members_1k
    for key in keys[1:]:
        common_members_1k = pd.merge(common_members_1k, storage[key][['source_id']], on='source_id', how='inner')

    ax1.axhline(len(common_members_1k), ls="-.", c="red", lw=1)

    # Calculate the y-coordinate for the text (assuming it's a horizontal line)
    y_coord = len(common_members_1k)

    ax1.set_ylim(round(y_coord-30), round(y_coord + 90))

    # Add the text just below the line without overlapping
    ax1.text(5.5, y_coord - 2 * ax2.get_ylim()[1] , f'Number of common members = {len(common_members_1k)}', color='black', ha='center', va='top', fontsize=20)

    # Annotate the data points with 'y' values
    for i, j in enumerate(y):
        ax1.text(i+1, j+3, str(j), ha='center', va='bottom',fontsize = 15)

    #################################################################################################################

    no_of_iterations = 100000
    storage_100k = {}
    y_10k = []

    for i in range(len(membership_record_100k)):
        
        hist, bin_edges = np.histogram(membership_record_100k[i], bins=no_of_bins, range=(0, no_of_iterations), density=True)
        
        # Create KDE
        kde = gaussian_kde(membership_record_100k[i], bw_method= bw)

        # Evaluate the KDE on a grid
        x_grid = np.linspace(0, no_of_iterations, 100000)
        y_kde = kde(x_grid)

        # Find minima using the scipy.signal.find_peaks function
        peaks, _ = find_peaks(y_kde)

        y_10k.append(len(membership_record_100k[i][membership_record_100k[i] > MC_thr]))
        
        # Store data in the dictionary
        storage_100k[f'iter_{i}'] = data[membership_record_100k[i] > MC_thr]

    ax2.plot(range(1,11),y_10k,linestyle ='-.',marker= 's',color = 'black' , markerfacecolor ='blue', markersize = 5)

    ax2.set_xlabel('No. of algorithm runs',fontsize = 22)
    ax2.set_ylabel('Member counts', fontsize = 22)
    # Set the main title and subtitle
    # Set the main title
    main_title = f'HDBSCAN-MC Stability Test: Cluster {cluster_no} [W4/W5 complex]'
    plt.suptitle(main_title, fontsize=24, y=1.01)

    # Add a subtitle below the main title
    subtitle = f'Iterations: 100,000'
    ax2.annotate(subtitle, (0.5, 1.015), xycoords='axes fraction', fontsize=22, ha='center')



    ax2.yaxis.set_major_locator(MultipleLocator(20))
    ax2.yaxis.set_major_formatter('{x:.1f}')
    ax2.yaxis.set_minor_locator(MultipleLocator(10))

    ax2.tick_params(axis='both',which='minor',direction='in' ,width=0.75,length=4,bottom=True, top=True ,left=True, right=True, labelsize = 13)
    ax2.tick_params(axis='both',which='major',direction='in',  width=1,length=7,bottom=True, top=True ,left=True, right=True, labelsize = 15)

    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_major_formatter('{x:.0f}')
    # ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax2.tick_params(axis='both',which='minor',direction='in' ,width=0.75,length=4,bottom=True, top=True ,left=True, right=True)
    ax2.tick_params(axis='both',which='major',direction='in',  width=1,length=7,bottom=True, top=True ,left=True, right=True)
    ax2.annotate('(b)', xy = annotation_coords, xycoords="axes fraction", fontsize=24, fontweight='normal')

    # Assuming storage is a dictionary containing DataFrames with keys iter_1 to iter_10
    # For example, storage = {'iter_1': df1, 'iter_2': df2, ..., 'iter_10': df10}

    # Get the keys from storage corresponding to iter_1 to iter_10
    keys = ['iter_' + str(i) for i in range(0, 10)]

    # Initialize common_members with the 'source_id' column from the first iteration
    common_members_100k = storage_100k[keys[0]][['source_id']]

    # Iterate over the keys and merge each 'source_id' column with common_members
    for key in keys[1:]:
        common_members_100k = pd.merge(common_members_100k, storage_100k[key][['source_id']], on='source_id', how='inner')

    ax2.axhline(len(common_members_100k), ls="-.", c="red", lw=1)


    # Calculate the y-coordinate for the text (assuming it's a horizontal line)
    y_coord = len(common_members_1k)
    # y_coord = len(common_members_100k)
    ax2.set_ylim(round(y_coord-30), round(y_coord + 90))

    y_coord = len(common_members_100k)
    # Add the text just below the line without overlapping
    ax2.text(5.5, y_coord - 0.01 * ax2.get_ylim()[1], f'Number of common members = {len(common_members_100k)}', color='black', ha='center', va='top', fontsize=20)

    # Annotate the data points with 'y' values
    for i, j in enumerate(y_10k):
        ax2.text(i+1, j+3, str(j), ha='center', va='bottom',fontsize = 15)

    
    plot_filename_stability = f"{cluster_name}_HDBSCAN-MC_stability_check.png"

    plt.savefig(plot_filename_stability, format = 'png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{cluster_name}_HDBSCAN-MC_stability_check.png", format = 'png', dpi=300, bbox_inches='tight')



    #==================================================Membership Distribution Plot===================================
    print("Plotting sample membership distribution...........")

    no_of_iterations = 100000
    no_of_bins = 100
    cut = 0.75
    bw = 0.05
    threshold_percentage = 100
    mem_record = membership_record_100k[5]
    hist, bin_edges = np.histogram(mem_record, bins=no_of_bins, range=(0, no_of_iterations),density = False)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax1 = plt.subplots(figsize=(8, 6))
    _ , bins, _ = ax1.hist(mem_record, bins=no_of_bins, range=(0, no_of_iterations), density=False, color='black', histtype='step', linewidth=2)

    # Create KDE
    kde = gaussian_kde(mem_record, bw_method = bw)

    # Evaluate the KDE on a grid
    x_grid = np.linspace(0, no_of_iterations, 10000)
    bin_width = bins[1] - bins[0]
    y_kde = kde(x_grid) * bin_width * len(mem_record)
    peaks, _ = find_peaks(y_kde)
    print(f"The MC_thr is : {MC_thr}")

    # Plot threshold line on the primary axis
    ax1.axvline(MC_thr, ls="-.", c="g", lw=2)

    ax1.set_xlabel("No of times YSOs recorded as cluster members", fontsize=20)
    ax1.set_ylabel("Number of YSOs", fontsize=20, color='black')
    ax1.set_title(f"Cluster {cluster_no} [W4/W5 complex]\n Membership distribution", fontsize=24)

    # Set axis limits
    ax1.set_xlim(0, no_of_iterations)

    # Add legend
    # ax1.legend(loc='upper left')

    # Add shaded area to the left of the vertical line
    ax1.axvspan(MC_thr,no_of_iterations, color='lightpink', alpha=0.5)

    # Adjust tick positions
    ax1.yaxis.set_major_locator(MultipleLocator(4))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.yaxis.set_minor_locator(MultipleLocator(2))

    ax1.xaxis.set_major_locator(MultipleLocator(no_of_iterations / 10))
    ax1.xaxis.set_major_formatter('{x:.0f}')
    ax1.xaxis.set_minor_locator(MultipleLocator(no_of_iterations / 20))

    # Set tick parameters
    ax1.tick_params(axis='both', which='minor', direction='in', width=0.75, length=3, bottom=True, top=True, left=True,
                    right=True, labelsize = 13)
    ax1.tick_params(axis='both', which='major', direction='in', width=1, length=7, bottom=True, top=True, left=True,
                    right=True, labelsize = 15)

    # Define the formatter function
    def thousands_formatter(x, pos):
        if x == 0:
            return ''  # Skip the 0th label
        else:
            return f'{int(x/1000)}k'

    # Apply the formatter to the x-axis
    ax1.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    # Remove gridlines
    ax1.grid(False)

    # Create secondary y-axis for KDE
    ax2 = ax1.twinx()

    ax2.plot(x_grid, y_kde, label='Scaled guassian kernel density estimate', color='firebrick', lw = 2)

    # Set ax2 y-limits to match ax1
    ax2.set_ylim(ax1.get_ylim())

    # Adjust tick positions
    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))
    ax2.set_ylabel("Scaled guassian kernel density estimate", fontsize=17, color='firebrick')

    ax2.yaxis.set_major_locator(MultipleLocator(4))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.yaxis.set_minor_locator(MultipleLocator(2))

    ax2.tick_params(axis='both', which='minor', direction='in', width=0.75, length=3, bottom=True, top=True, left=True,
                    right=True, labelsize = 13)
    ax2.tick_params(axis='both', which='major', direction='in', width=1, length=7, bottom=True, top=True, left=True,
                    right=True, labelsize = 15)
    
    
     # Define the formatter function
    def ax2_formatter(y, pos):
        if y == 0:
            return ''  # Skip the 0th label
        else:
            return int(y)
    # Apply the formatter to the x-axis
    ax2.yaxis.set_major_formatter(FuncFormatter(ax2_formatter))
    plt.tight_layout()


    ################################################################################################
    plot_filename_mem = f"Membership_distribution_{cluster_name}_histonly.png"
    # plot_filename_mem = f"{cluster_name}/"+ plot_filename
    plt.savefig(plot_filename_mem, format ='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"Membership_distribution_{cluster_name}_histonly.png", format ='png', dpi=300, bbox_inches='tight')
    plt.close()
    # ###############################################################################################

    print(f"MC Threshold is: {MC_thr}")
    print(f"Number of iterations above threshold: {len(mem_record[mem_record > MC_thr])}")

    #==================================================Astrometric Analysis Plot===================================
    print("Generating astrometric plot for the cluster...........")

    # Plotting the Cluster members after Monte Carlo Sampling
    cluster_labels_copy = common_members_100k

    cluster_data = data
    annotation_coords = (-0.12 , 1.02)
    fig, ax = plt.subplots(2, 3, figsize=(27,15))
    fig.suptitle(f"Cluster {cluster_no}: Monte-Carlo Simulation + HDBSCAN*", fontsize=27)

    # Adjust layout for title
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    MC_thr = MC_thr 

    ax[0, 0].invert_xaxis()

    plot_2d('ra_epoch2000_x', 'dec_epoch2000_x', ax[0,0], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr, leg_flag = 0, alpha1=1, alpha2 = 1)
    ax[0,0].annotate('(a)', xy = annotation_coords, xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[0,0].set_xlabel('RA in degrees (J2000) ', fontsize = 20)
    ax[0,0].set_ylabel('Dec in degrees (J2000)',fontsize = 20)
    ax[0,0].set_title('RA - Dec Space',fontsize = 22)

    ax[0,0].scatter(ir_table['RA_deg'], ir_table['Dec_deg'], alpha = 1 ,s=10, c=gcolor[2], marker=gmarker[2],label = 'SFOG YSOs')
    print(f'Total SFOG only members: {len(ir_table)}')
    ax[0,0].legend(loc='upper right', bbox_to_anchor=(3.5, 1.37),fontsize = 17)

    # plot_2d('ra_epoch2000_x', 'dec_epoch2000_x', ax[0,0], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr,leg_flag = 0,alpha1=1,alpha2 = 1)
    # ax[0,0].annotate('(a)', xy = annotation_coords, xycoords="axes fraction", fontsize=20, fontweight='normal')
    # ax[0,0].set_xlabel('longitude (l) in degrees (J2016)',fontsize = 20)
    # ax[0,0].set_ylabel('latitude (b) in degrees (J2016)',fontsize = 20)
    # ax[0,0].set_title('l - b Space',fontsize = 22)

    plot_2d('pmra', 'pmdec', ax[0,1], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr ,leg_flag =0,alpha1=1,alpha2 = 1)
    ax[0,1].annotate('(b)', xy = annotation_coords , xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[0,1].set_xlabel('Proper motion along RA (mas/yr) ',fontsize = 20)
    ax[0,1].set_ylabel('Proper motion along Dec (mas/yr)',fontsize = 20)
    ax[0,1].set_title('Proper Motion Vector Plot (J2016)',fontsize = 22)


    plot_2d('g_rp','bp_g', ax[0,2], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr ,leg_flag = 0,alpha1=1,alpha2 = 1)
    ax[0,2].annotate('(c)', xy = annotation_coords , xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[0,2].set_xlabel(r'$G - G_{RP}$',fontsize = 20)
    ax[0,2].set_ylabel(r'$G_{BP} - G$',fontsize = 20)
    ax[0,2].set_title('Gaia Color-Color Diagram',fontsize = 22)


    plot_2d('X', 'Y', ax[1,0], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr ,leg_flag = 0,alpha1=1,alpha2 = 0.5)
    ax[1,0].annotate('(d)', xy = annotation_coords , xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[1,0].set_xlabel('X (pc)',fontsize = 20)
    ax[1,0].set_ylabel('Y (pc)',fontsize = 20)
    ax[1,0].set_title('X - Y Projection (J2016)',fontsize = 22)

    plot_2d('X', 'Z',ax[1,1], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr ,leg_flag = 0,alpha1=1,alpha2 = 0.5)
    ax[1,1].annotate('(e)', xy = annotation_coords , xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[1,1].set_xlabel('X (pc)',fontsize = 20)
    ax[1,1].set_ylabel('Z (pc)',fontsize = 20)
    ax[1,1].set_title('X - Z Projection (J2016)',fontsize = 22)

    plot_2d('Y', 'Z', ax[1,2], cluster_data, cluster_labels_copy, MCflag = 1, MC_thr = MC_thr ,leg_flag = 0, alpha1=1,alpha2 = 0.5)
    ax[1,2].annotate('(f)', xy = annotation_coords , xycoords="axes fraction", fontsize=20, fontweight='normal')
    ax[1,2].set_xlabel('Y (pc)',fontsize = 20)
    ax[1,2].set_ylabel('Z (pc)',fontsize = 20)
    ax[1,2].set_title('Y - Z Projection (J2016)',fontsize = 22)

    # plt.tight_layout()
    plot_filename_astro = f"Astrometric_analysis_{cluster_name}.png"
    plt.savefig(plot_filename_astro, dpi=300)

    #==================================================Calculating Cluster Statistics========================
    print("Calculating cluster statistics...........")

    # ind2 =  data['source_id'].isin(common_members_100k['source_id'])
    ind2 = data['mem_flag']==1
    print(f" Median Distance {np.median(data[ind2]['final_dist'])}")
    median_distance = np.median(data[ind2]['final_dist'])

    print(f" Mean Distance {np.mean(data[ind2]['final_dist'])}" )
    mean_distance = np.mean(data[ind2]['final_dist'])

    print(f"Average distance uncertainity {np.mean( (data[ind2]['final_dist_hi'] - data[ind2]['final_dist_lo'] )/2 )}")
    avg_dist_uncer = np.mean( (data[ind2]['final_dist_hi'] - data[ind2]['final_dist_lo'] )/2 )

    #==================================================Saving Member Files===================================
    print("Saving detected cluster members to files...........")

    t_IR = Table.from_pandas(data[ind2][~(data[ind2]['ID'].astype(str)== 'nan')])
    t_IR_total = data[~(data['ID'].astype(str)== 'nan')]
    output_directory_IR = f"{cluster_name}_IR.fits"
    t_IR.write(output_directory_IR, format='fits', overwrite=True)

    t_optical = Table.from_pandas(data[ind2][(data[ind2]['ID'].astype(str)== 'nan')])
    t_optical_total = data[(data['ID'].astype(str)== 'nan')]
    output_directory_optical = f"{cluster_name}_optical.fits"
    t_optical.write(output_directory_optical, format='fits', overwrite=True)

    t_total = Table.from_pandas(data[ind2])
    output_directory_total = f"{cluster_name}_all_members.fits"
    t_total.write(output_directory_total, format='fits', overwrite=True)

    t_IR_non_members = Table.from_pandas(data[~ind2][~(data[~ind2]['ID'].astype(str)== 'nan')])

    # Assuming 't_IR_non_members' and 'sfog_cluster_data' are Astropy tables
    reject_for_beta = join(t_IR_non_members[['ID','IR name']], sfog_cluster_data, keys='ID', join_type='inner', table_names=['t_IR_non_members', 'sfog_cluster_data'])
    # intersection_table now contains rows that have common 'ID' values in both tables,
    # with columns only from sfog_cluster_data

    ## Uncomment the lines below if the dataframe 'reject_for_beta' needs to written out as a FITS file.
    reject_for_beta.write('reject_for_beta.fits', format = 'fits' , overwrite = True)

    beta_reject = len(reject_for_beta)
    false_positives = MC_table['f_MC_thr'][0]
    true_positives = MC_table['true_MC_thr'][0]

    #============================================Table for optimization solutions===================================

    params_dir = directory+ f"/simulated_clusters/Data/Cluster_{cluster}_sim/parameters_Cluster_{cluster}.txt"
    params_table = pd.read_csv(params_dir, delimiter='\t')

    params_main = params_table[params_table['residual'] != -1]
    min_residual = params_main['residual'].min()
    params_imp = params_main[params_table['residual'] <= 3 * min_residual]

    # Sort the filtered table based on the 'residual' column
    sorted_params = params_imp.sort_values(by='residual').round(4)

    # Construct the new DataFrame
    first_row_data = np.round(params_table.iloc[0].values.astype(float),decimals = 3)

    second_row_data = np.round(params_table.iloc[1].values.astype(float),decimals = 3)

    new_first_row = []
    for i, (val, uncertainty) in enumerate(zip(first_row_data, second_row_data)):
        if i == 1:  # Index of the second value
            new_first_row.append(f'{round(val)} pc +/- {round(uncertainty)} pc')
            sim_dist = round(val)
            sim_dist_unc = round(uncertainty)
        else:
            new_first_row.append(f'{round(val, 3)} +/- {round(uncertainty, 3)}')

    new_first_row[-1] = ' '
    new_first_row[0] = ' '
    # Drop the first two rows from the sorted table
    final_table = sorted_params.copy()

    # Create DataFrame from new_first_row list
    new_first_row_df = pd.DataFrame([new_first_row], columns=final_table.columns)

    # Concatenate new_first_row_df with final_table
    final_table = pd.concat([new_first_row_df, final_table], ignore_index=True)


    # Assuming your DataFrame is named final_table
    final_table = final_table.rename(columns={final_table.columns[0]: 'Iteration'})
    final_table = final_table.rename(columns={final_table.columns[7]: 'MSD'})

    final_table.to_csv('chosen_parameters.txt', sep = ' ', index=False)

    #======================================Saving statistical data to text file=================================

    kinematic = Table.read("../DATA/kinematic.fits")
    # kinematic = kinematic.to_pandas()

    if np.sum(kinematic['Cluster #'].astype(int) == cluster_no) == 0:
        print("Kinematic distance estimate not available")
        kinematic_distance = "N/A"
        kinematic_unc = ''
    else:
        kinematic_distance = round(float(kinematic['Distance kpc'][kinematic['Cluster #'].astype(int) == cluster_no][0])*1000)
        kinematic_unc = round(0.2*kinematic_distance)
    beta = round( (len(t_total) - false_positives) / (IR_no - beta_reject) , 4)
    # beta = round( (true_positives) / (IR_no - beta_reject) , 4)

    t_IR_pandas = t_IR.to_pandas()
    sfog_cluster_data_pandas = sfog_cluster_data.to_pandas()

    # t_new_IR represents new IR members in SFOG matched Gaia not in sfog cluster member consideration
    t_new_IR = t_IR_pandas[~t_IR_pandas['ID'].isin(sfog_cluster_data_pandas['ID'])] 

    # Define the content to write to the file
    content = (
        f"cluster_name, central_RA, central_Dec, cluster_radius, DBSCAN, HDBSCAN_MC, total_HDBSCAN_MC, new_gaia, total_gaia, new_SFOG,  outliers_DBSCAN, beta, median_dist, mean_dist, avg_dist_unc, kinematic_dist, kin_unc, sim_dist, sim_dist_unc, cen_lon, cen_lat, f_pos \n"  
        f"{cluster_name}, {round(cen_RA,5)}, {round(cen_Dec,5)}, {round(cluster_radius,5)}, {IR_no}, {len(t_IR)}, {len(t_IR_total)}, {len(t_optical)}, {len(t_optical_total)}, {len(t_new_IR)}, {beta_reject}, {beta}, {round(median_distance)}, {round(mean_distance)}, {round(avg_dist_uncer)}, {kinematic_distance}, {kinematic_unc}, {sim_dist}, {sim_dist_unc} ,{round(cen_lon,5)}, {round(cen_lat,5)}, {false_positives}")

    # Specify the file path
    text_file_path = f"{cluster_name}.txt"

    # Write the content to the file
    with open(text_file_path, "w") as file:
        file.write(content)

    print("Cluster statistics written to the file.")

    #============================================Proper Motion Analysis===================================
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_pm_histogram(angle, angle_min, angle_max, num_bins=20):
        """
        Plots a histogram of distances and calculates peak value, 1-sigma, and 3-sigma deviations.

        Parameters:
        distances (array-like): Array containing distance values.
        num_bins (int, optional): Number of bins in the histogram. Default is 20.
        show_plot (bool, optional): Whether to display the histogram plot. Default is True.

        Returns:
        peak (float): Peak value of the histogram.
        sigma_1 (float): 1-sigma deviation from the peak.
        sigma_3 (float): 3-sigma deviation from the peak.
        """

        fig, ax = plt.subplots()
        # Create a histogram of distances and collect histogram data
        counts_mean, bin_edges_mean, _ = ax.hist(angle, bins=num_bins, density=False, alpha=0.5, color='maroon', label="Mean")
        counts_min, bin_edges_min, _ = ax.hist(angle_min, histtype='step', bins=num_bins, density=False, alpha=1, linestyle='--', linewidth=1.3, edgecolor='black', label="Minimum Deviation")
        counts_max, bin_edges_max, _ = ax.hist(angle_max, histtype='step', bins=num_bins, density=False, alpha=1, linestyle='-', linewidth=1.5, edgecolor='#080da3', label="Maximum Deviation")


        # Find the bin with the highest count (peak)
        peak_bin = np.argmax(counts_mean)
        peak_value = bin_edges_mean[peak_bin]

        # Calculate the standard deviation of the distances
        sigma = np.std(angle)

        # Calculate 1-sigma and 3-sigma deviations from the peak
        sigma_1 = peak_value + sigma
        sigma_3 = peak_value + 3 * sigma

        return ax,peak_value, sigma_1, sigma_3

    pmra = data[ind2]['pmra']
    pmdec = data[ind2]['pmdec']
    angle_mean = np.degrees(np.arctan2(pmdec, pmra))

    # Deviation 1- Maximum attempted
    data['pmra_min'] =  data['pmra'] - np.multiply(np.sign(data['pmra']),data['pmra_error'])
    pmra_min = data[ind2]['pmra_min'] 
    data['pmdec_max'] =  data['pmdec'] + np.multiply(np.sign(data['pmdec']),data['pmdec_error'])
    pmdec_max = data[ind2]['pmdec_max'] 

    angle_1 = np.degrees(np.arctan2(pmdec_max, pmra_min))


    # Deviation 2- Minimum attempted
    data['pmra_max'] =  data['pmra'] + np.multiply(np.sign(data['pmra']),data['pmra_error'])
    pmra_max = data[ind2]['pmra_max'] 
    data['pmdec_min'] =  data['pmdec'] - np.multiply(np.sign(data['pmdec']),data['pmdec_error'])
    pmdec_min = data[ind2]['pmdec_min'] 

    angle_2 = np.degrees(np.arctan2(pmdec_min, pmra_max))

    def swap_angles(angle_1, angle_2):
        # Ensure both angle_1 and angle_2 are pandas Series
        """
        Swap elements of two angle Series so that angle_1 contains the larger absolute angle values.

        Parameters
        ----------
        angle_1 : pandas Series
            First angle Series.
        angle_2 : pandas Series
            Second angle Series.

        Returns
        -------
        angle_1 : pandas Series
            First angle Series with swapped elements.
        angle_2 : pandas Series
            Second angle Series with swapped elements.
        """
        if not isinstance(angle_1, pd.Series) or not isinstance(angle_2, pd.Series):
            raise ValueError("Both angle_1 and angle_2 must be pandas Series.")

        # Convert Series to lists for modification
        angle_1_list = angle_1.tolist()
        angle_2_list = angle_2.tolist()

        # Compare absolute values of angles and swap if necessary
        for i in range(len(angle_1_list)):
            if abs(angle_1_list[i]) <= abs(angle_2_list[i]):
                angle_1_list[i], angle_2_list[i] = angle_2_list[i], angle_1_list[i]

        # Convert lists back to Series
        angle_1 = pd.Series(angle_1_list)
        angle_2 = pd.Series(angle_2_list)

        return angle_1, angle_2

    angle_max, angle_min = swap_angles(angle_1, angle_2)

    ax,_,_,_ = plot_pm_histogram(angle_mean, angle_min, angle_max, num_bins = 50)



    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, MaxNLocator,AutoMinorLocator, MultipleLocator, FuncFormatter

    # Set tick parameters
    ax.tick_params(axis='both', which='minor', direction='in', width=0.75, length=3, bottom=True, top=True, left=True,
                    right=True)
    ax.tick_params(axis='both', which='major', direction='in', width=1, length=7, bottom=True, top=True, left=True,
                    right=True)

    ax.set_title(f'Proper Motion Direction Distribution for Cluster {cluster_no}')
    ax.set_xlabel('Angle ( measured from positive x-axis in degrees)')
    ax.set_ylabel('Frequency')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.gca().xaxis.set_major_locator(MaxNLocator(15))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

    plt.xticks(fontsize=8)

    xlimits = ax.get_xlim()

    if xlimits[1]-xlimits[0]<5:
        plt.gca().xaxis.set_major_locator(MaxNLocator(11))
        plt.xticks(fontsize=8.5)

    plt.legend()
    plot_filename_pmangle = f"pmangle_analysis_{cluster_name}.png"
    plt.savefig(plot_filename_pmangle, dpi=300)

    #============================ Now time for magnitude plot=================================

    x = data['pmra']
    y = data['pmdec']

    x_err = data['pmra_error']
    y_err = data['pmdec_error']

    mag_err = np.abs(np.divide(x*x_err, np.sqrt(x**2 + y**2))) + np.abs(np.divide(y*y_err, np.sqrt(x**2 + y**2)))

    mag = data[ind2]['pm']
    mag_err = mag_err[ind2]

    ax1,_,_,_ = plot_pm_histogram(mag, mag-mag_err, mag+mag_err, num_bins = 50)

    # Set tick parameters
    ax1.tick_params(axis='both', which='minor', direction='in', width=0.75, length=3, bottom=True, top=True, left=True,
                    right=True)
    ax1.tick_params(axis='both', which='major', direction='in', width=1, length=7, bottom=True, top=True, left=True,
                    right=True)

    ax1.set_title(f'Proper Motion Magnitude Distribution for Cluster {cluster_no}')
    ax1.set_xlabel('Proper motion magnitude (mas/year)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.7)

    plt.gca().xaxis.set_major_locator(MaxNLocator(11))
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

    plt.xticks(fontsize=9)

    plt.legend()
    plot_filename_pm_mag = f"pm_mag_analysis_{cluster_name}.png"
    plt.savefig(plot_filename_pm_mag, dpi=300)

    # #==================================================Generating Cluster Report===================================
    from pylatex import Document, Section, Subsection, Command , Table, Tabular , Figure , MultiColumn , LongTable, SmallText
    from pylatex.utils import italic, NoEscape , bold

    # Create a Document object
    doc = Document(geometry_options=["left=2cm", "right=2.5cm", "bottom=1.5cm", "top=0.8cm"])

    # Add a title to the document
    cluster_no = int(cluster_name.split('_')[1])
    doc.preamble.append(Command('title', f'Cluster {cluster_no}'))
    doc.append(NoEscape(r'\maketitle'))

    # Add a section
    with doc.create(Section('Identifying Genuine cluster members with 3D Monte-Carlo simulation and HDBSCAN')):
        doc.append(f'This is an auto generated report for Cluster {cluster_no}. \n \n')
        doc.append('It shows methodology for identifying genuine cluster memebers with 3D Monte-Carlo Simulation and HDBSCAN*. Similar reports have been generated for all analyzed clusters.')


    # Add a subsection
    with doc.create(Subsection('Determination of membership cut-off from membership distribution')):
        with doc.create(Figure(position='h')) as mem_figure:
            mem_figure.add_image(plot_filename_mem, width='300px')
            mem_figure.add_caption(f'Distribution of YSO membership for 100,000 iterations of')
            mem_figure.append(f'HDBSCAN* for {cluster_no}')

        doc.append(NoEscape(r'\noindent'))
        doc.append(f'Monte-Carlo threshold (green dash-dotted line): {MC_thr}\n')
        # doc.append(f'Minima is at (blue dash-dotted line) : {MC_thr_min}\n')
        doc.append(f'Number of members above threshold: {len(mem_record[mem_record > MC_thr])}')
        

    # Add a subsection
    doc.append(NoEscape(r'\hfill'))
    with doc.create(Subsection('Identifying common members across 10 runs HDBSCAN-MC')):
        doc.append('Figure 2 shows stability of HDBSCAN-MC across 10 runs. The common members across all runs will be considered for further analysis')
        
        # # Add text
        # doc.append('The angle of the proper motion vector is calculated using numpy.')
        
        with doc.create(Figure(position='h')) as stab_figure:
            stab_figure.add_image(plot_filename_stability, width='410px')
            stab_figure.add_caption(f'Stability Test for identification of HDBSCAN-MC algorithm ')
            stab_figure.append(f'across 10 runs. Common members across various runs are taken as final members')

    doc.append(NoEscape(r'\newpage'))
    # Add a subsection
    with doc.create(Subsection('Viewing identified genuine cluster members in astrometric space.')):
        doc.append('Figure 3 shows final identified members in astrometric space along with Gaia color-color diagram.')
        
        with doc.create(Figure(position='h')) as stab_figure:
            stab_figure.add_image(plot_filename_astro, width='535px')
            stab_figure.add_caption(f'Astrometric analysis by application of HDBSCAN-MC')
            stab_figure.append(f'for Cluster {cluster_no}')

    doc.append(NoEscape(r'\newpage'))
    # Add a table
    with doc.create(Subsection('Summarizing Cluster Statistics')):
        
        with doc.create(Table(position='h')) as table:
            with table.create(LongTable('|c|c|')) as tabular:
                tabular.add_hline()
                tabular.add_row([MultiColumn(2, align='|c|', data=bold('Cluster Statistics'))])
                tabular.add_hline()
                tabular.add_row("Median Distance", str(int(median_distance))+ ' pc +/- ' + str(int(avg_dist_uncer)) + ' pc')
                tabular.add_row("Mean Distance",  str(int(mean_distance)) + ' pc +/- ' + str(int(avg_dist_uncer)) + ' pc')
                # tabular.add_row("Average Distance Uncertainty",  str(int(avg_dist_uncer)) + ' pc')
                tabular.add_row(" "," ")
                if kinematic_distance == 'N/A':
                    tabular.add_row("Kinematic Distance (Winston et.al. 2020)", str(kinematic_distance))
                else:
                    tabular.add_row("Kinematic Distance (Winston et.al. 2020)", str(kinematic_distance) + ' pc +/- ' + str(kinematic_unc)+' pc')
                tabular.add_row("Simulation predicted distance estimate", final_table['center_dist'][0])
                tabular.add_row(" "," ") 
                tabular.add_row("No.of Gaia matched SFOG YSOs", f"{len(t_IR)} out of {len(t_IR_total)}")
                tabular.add_row("No.of Gaia only YSOs", f"{len(t_optical)} out of {len(t_optical_total)}")
                tabular.add_row("Total cluster members (with Gaia counterparts)", f"{len(t_total)} out of {len(data)}")
                tabular.add_row("Predicted false positives", f"{false_positives} out of {len(t_total)}")
                tabular.add_row("New Gaia-matches SFOG members", f"{len(t_new_IR)}")
                # tabular.add_row("Predicted true positives", f"{true_positives} out of {len(t_total)}")
                tabular.add_row("SFOG only YSOs in region", f"{len(ir_table)}")
                tabular.add_row("SFOG cluster members as reported in Winston et.al. (2020)", f"{IR_no}")
                tabular.add_row("No. of outlier in Winston et.al. (2020) SFOG cluster members", f"{beta_reject}")
                tabular.add_row(" "," ")
                tabular.add_row(NoEscape(r'Cluster $\beta$ value'), str(beta))
                tabular.add_hline()


    # doc.append(NoEscape(r'\newpage'))
    with doc.create(Subsection('Proper Motion Analysis')):

        doc.append('Figure 4 shows direction distribution of proper motion vectors for identified cluster members.' 
                ' The angle is measured from positive x-axis. The proper motion magnitude distribution is given by Figure 5.')

        with doc.create(Figure(position='h')) as pm_figure:
            pm_figure.add_image(plot_filename_pmangle, width='400px')
            pm_figure.add_caption(f'Proper motion direction distribution')
            pm_figure.append(f'for Cluster {cluster_no}')

        # doc.append('The proper motion magnitude distribution is given by Figure 5')

        with doc.create(Figure(position='h')) as pm_figure:
            pm_figure.add_image(plot_filename_pm_mag, width='400px')
            pm_figure.add_caption(f'Proper motion magnitude distribution')
            pm_figure.append(f'for Cluster {cluster_no}')

    doc.append(NoEscape(r'\newpage'))
    with doc.create(Subsection('Cluster simulation')):
        doc.append('The table below gives weighted parameters and unbiased weighted standarad deviation obtained from optimization process. The weighted parameters are calculted using optimized parameter sets having MSD less than three times best-fit solution. \n \n')
        

    # Create a pylatex tabular environment with smaller font size
    doc.append(NoEscape(r'\tiny'))
    with doc.create(Tabular('|c|c|c|c|c|c|c|c|')) as table:
        # Add column headers
        table.add_hline()
        table.add_row(final_table.columns)
        table.add_hline()
        # Add rows from the DataFrame
        for index, row in final_table.iterrows():
            table.add_row(row)
            table.add_hline()

    
    with doc.create(Figure(position='h')) as ms_figure:
        ms_figure.add_image(directory + f"/simulated_clusters/Cluster_{cluster}_sim/mspec_Cluster_{cluster}.png", width='400px')
        ms_figure.add_caption(f'Monte Carlo spectra and simulation result')
        ms_figure.append(f'for Cluster {cluster_no}')


    doc.generate_pdf(filepath=f"{cluster_name}",compiler='pdflatex')