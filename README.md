#  HDBSCAN-MC and GOLDFIST Simulation

## 📖 Project Description
The codebase applies the HDBSCAN* to Gaia DR3 astrometry for YSO cluster membership assignment. The HDBSCAN-MC method integrates Monte Carlo simulations to handle astrometric uncertainties in 5D clustering. The GOLDFIST simulation which standes for Generation Of cLuster anD FIeld STar simulation simulates cluster and field YSOs. The simulation enables determination of Monte-Carlo threshold which is required parameter for HDBSCAN-MC based cluster member determination and thereby enables robust, unsupervised membership determination while accounting for measurement errors.

HDBSCAN-MC, in conjunction with the GOLDFIST simulation, provides a robust framework for systematic cluster property analysis and membership determination, making it a valuable tool for future Galactic surveys. 

## 🛠 Installation

This section provides the required dependencies and installation instructions.

1. Clone the repository:
   ```sh
   git clone https://github.com/vpatel2000/HDBMC-GOLDFIST-Sim.git
2. Navigate to the project directory:
   ```sh
   cd HDBMC-GOLDFIST-Sim

The following python packages need to be installed:

1. numpy
2. pandas
3. matplotlib
4. scipy
5. astropy
6. configparser
7. importlib
8. os
9. gc
10. warnings

The code also requires access to a High Performance Cluster facility.\
In this study we have used Smithsonian Institution Hydra cluster.

## 🚀 Directory structure & Usage

We have included the data and results of the three analyzed cluster in our study. Here we show how this code is used for Cluster 257 same process can be used for any other cluster. 

### Directory Tour

#### 🔹 Observed Cluster Data  
- The observed cluster data and results are stored in the **`Cluster_257`** folder.  
- The **`DATA`** folder contains the Gaia-SFOG crossmatch dataset and other required data files.  
- To set up the project directory, specify the main local directory path in `config.ini` as:  
  ```ini
   primary_dir = /your/local/path/
  ```
  This ensures all paths in other Python scripts update accordingly.

- The `optimize_gen_257.job` file contains details to execute the `otimize_gen_257.py` file on Hydra HPC cluster. The output of this file is logged in `optimize_gen_257.log` file

#### 📊 Cluster Report Generation

- The script cluster_report_generator.py can be used generate `cluster_257.pdf` which summarizes the final results for the observed cluster. 
- The pdf file can be found in the `Cluster_257` folder.

#### 🔹 Simulated Cluster Data
The simulated cluster results and generated data are present in `simulated_clusters folder.`A few imprtant files in this folder are as follows:

- The simulated cluster data are present `Data/Cluster_257_sim`. with required_sim_data_Cluster_257.fits giving the final simulated cluster with inculcated distance uncertainities in the simulation.

- In this Data folder `more_real_pm_cut_2.fits` gives The Taurus-Auriga complex data required for proper motion optimization.

#### 🛠️ Key Scripts & Functions

- The `cluster_model.py` consists of functions required for cluster simulation (currently using King profile), assigns astrometric data and errors to the simulated field and cluster members YSOs. [Primary file for GOLDFIST simulation]

- `MS_engine.py` generates the Monte Carlo spectrum. It also conatins function to obtain weighted average and weighted standarad deviations.

- `HDBSCAN_func.py` consists of list of functions required for HDBSCAN_MC implementation and subsequent analysis.

#### 🚀 Running the Optimization

The user primarily interacts with `optimize_gen_257.py`, which performs:  

- Two-stage global optimization (spatial and proper motion) using differential evolution.  
- Application of HDBSCAN-MC on the optimized simulated cluster to derive the Monte Carlo threshold.  
- Membership determination in the observed cluster using the derived threshold.  

### Usage

Before running the optimization, update the following files as explained in the previous subsection:  

- **`config.ini`** – Make necessary path changes. Modify config_path in cluster_model.py 
   ```python
   config_path = "/your/local/path/config.ini" 
   ``` 
- **`optimize_gen_257.job`** – Adjust job submission settings for the HPC environment like number of used CPU cores.  
- **`optimize_gen_257.py`** – Apply any required changes based on your setup. Any change in number of CPU cores also needs to be updated here.  

Run the following commands in the terminal:  

```bash
cd Cluster_257  
qsub optimize_gen_257.job  
```

The optimization process executed by `optimize_gen_257.py` file can be tracked in `optimize.gen_257.log` file.

### ⚠️ Repository Status  

This repository is currently archived. However, we will welcome contributions in the future once the project is open for updates!  

### 🤝 Future Contributions  

If you’d like to contribute once the project is open to changes, please follow these steps:  

#### 1. Fork the repository  

   Click the "Fork" button at the top right of the repository page to create a copy of this repo under your GitHub account.

#### 2. Clone your forked repository
   ```sh
      git clone https://github.com/yourusername/repository-name.git
      cd repository-name
   ```   
#### 3. Create a new branch
   ```sh
   git checkout -b feature-branch
   ```
#### 4. Make your changes

   Implement your improvements or bug fixes and commit them:
   ```sh
   git add .
   git commit -m "Describe your changes"
   ```

#### 5. Push to your branch and submit a Pull Request.
   ```sh
   git push origin feature-branch
   ```
- Go to the original repository on GitHub.
- Click on "Pull Requests" > "New Pull Request".
- Select your forked branch and submit the PR with a clear description.

## 📌 Citation and BibTeX Entry (To be updated)  

If you use this code in your research, please cite our forthcoming paper:  

**Identification of Outer Galaxy Cluster Members Using Gaia DR3 and Multidimensional Simulation**  
Vishwas Patel, Joseph L. Hora, Matthew L.N. Ashby, Sarita Vig  
_To be submitted to The Astrophysical Journal_

Once published, the citation details will be updated here. The GitHub repository link is included in the manuscript.  

A BibTeX entry will be provided upon publication for easy citation.  

Thank you for citing our work! 
 
## 📜 License  
This project is licensed under the [MIT License](LICENSE).  
