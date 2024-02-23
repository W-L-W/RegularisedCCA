# RegularisedCCA

This repository contains the code for the illustrations in the preprint "...", which can be found [here](link-to-follow).

We provide the run scripts for obtaining the numerical results and plots in the paper in `expmts/input_scripts` (see [Running Experiments](#running-experiments)); this uses python code in the `src` that has been packaged as a python package (see [Package Structure](#package-structure)).

## Installation
### Summary
Most required packages are available on conda (pandas, jupyter, matplotlib, scikit-learn, statsmodels, pygraphviz, plotly); however the following two more specific packages need to be installed via pip:
- cca-zoo
- gglasso

Recent versions of these packages will hopefully be sufficient to run all the code in this repository; we found conda helpful to find a compatible set of these required packages. 
One possible set of versions is given in the `environment.yml` file, and used in the more detailed instructions in the next subsection, but users may also like to experiment with other tools for dependency management.


### Detail
1. Clone the repo
```
git clone https://github.com/W-L-W/RegularisedCCA.git
```

2. Create the `regularised-cca-env` environment from yml file 
```
conda env create -f environment.yml
```

3. Activate `regularised-cca-env` environment
```
conda activate regularised-cca-env
```

4. Pip install `src` as a package: navigate to the parent directory of `src` and run
```
pip install -e .
```
(the flag -e is for editable mode, which will allow you to make changes to the files in `src` without having to reinstall the package)


### Troubleshooting
I previously encountered a problem with my matplotlib version, that appeared to be fixed by running
```
pip install --upgrade --force-reinstall matplotlib
```

Please don't hesitate to reach out if you have any issues.

## Running experiments

We provide the run scripts for obtaining the numerical results and plots in the paper in `expmts/input_scripts`.
The associated output is generated to the `expmts/output` directory.
Plots and lightweight summaries are available in the `plots` and `processed` subdirectories.
Estimated weight matrices are all cached to `expmts/output/detail` when the fitting scripts are run; these cached files are numerous and large, so the `detail` subdirectory is git-ignored.


## Data
We include csv files of all the raw data used for these experiments in `real_data`, along with utility functions for loading data matrices and variable labels in standardised format in `real_data.loading.py`.


## Package structure
To avoid circular import errors, we have structured the files in this project to satisfy the following topological ordering (no dependencies at top, semi-colons in brackets indicate left-to-right dependencies, commas indicate that there are no dependencies between adjacent files):

utils (linalg; cca, covs)  
algos (gCCA, sCCA, sPLS; combined)  
scaffold (io_preferences; core)  
real_data (loading; styling)  
scaffold (wrappers; synthetic)  
plots (basic; all-other-plots)  

| Directory | Files | Description |
| --------- | ----- | ----------- |
| utils | linalg; cca, covs | basic utility functions for linear algebra and canonical correlation analysis |
| Directory | Files | Description |
| --------- | ----- | ----------- |
| utils | linalg; cca, covs | Basic utility functions for linear algebra and canonical correlation analysis |
| algos | gCCA, sCCA, sPLS; combined | Algorithms for graphical CCA, sparse CCA, sparse PLS, and unified interface |
| scaffold | io_preferences; core | Input/output preferences and core functionality of caching framework for comparison of different estimates |
| real_data | loading; styling | For loading data matrices and variable labels in standardized format (uses objects from `scaffold.core`) |
| scaffold | wrappers; synthetic | Wrapper functions exploiting the scaffold caching framework, and synthetic data generation |
| plots | basic; ... | Plotting functions |
