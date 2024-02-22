# Next tasks:
- Nutrimouse pbootstrap fit and oracle panel recover
    Have now successfully created a numpy array for the parametric bootstrap nutrimouse
    Next to copy and paste over the oracle plots (make sure they exist)
    Then to fit the model and see if the oracle plots render
    This will require figuring out how to save plots effectively
    And further experimentation with gitignore template matching

# Later tasks:
- Add a test that creates synthetic covariance and check that CCA function recovers desired structure
- What are the different types of plot that need to be fitted into the new framework?

# Wed 14th Feb; Planning
The notes above do seem reasonable. Review the code base and get a better feel about how to implement these.
Checked with the results in the document and cvm3 is indeed CV minus factor of 3 as hoped.
So now need code to fit / run...
Important material seems to be in oop_utils; compute everything and setting
Some untidiness of distribution of code between wrappers and synthetic - will need to review that at some point but worth cracking on for now to get a better idea of what to deal with

Recall that for now want to produce minimal quantity of plots required for essay.  
This is what is required for the `essay.ipynb` file; so can work through that!
Yes, got the assertion error that I was hoping for. Happy days.

# Recall formatting for load df summary
Changed load_dffull to load_df_oracle's for MVNSolPath and MVNCV
Hopefully will all work now?




# Tuesday 20 Feb:
For return on Wednesday - check that the new file structure has not broken the saving.py content then commit, push, then keep going through `essay.ipynb` for nutrimouse in that script

# New experiment folder structure
Does actually make sense I think.
Idea is that there will be a small number of scripts in `expmts/in` that can be run from the command line
Some of these will be of the flavour 'compute everything' and create many files in `expmts/out/detail`
Others will be of the flavour 'create certain plots'
There will be a lot of plots, but a manageable number, and for convenience of putting into the overleaf file, I think it is in fact OK to have these in a single folder
# A topological ordering
Of the files in this project to prevent any circular import errors
(no dependencies at top)
utils (linalg; cca, covs)
algos (gCCA, sCCA, sPLS)
scaffold (incoming; core)
real_data (loading; styling)
scaffold (wrappers; synthetic)
plots (basic; the_rest)

Extra notes:
This is one possible order, there are many other options also (e.g. real_data.styling has no dependnecies, it only needs np and pd)



# Some-day maybe
- Fun reading: https://medium.com/brexeng/avoiding-circular-imports-in-python-7c35ec8145ed

# Archival Notes
## Plot classification
Data Histogram
Bubble plots and derivatives
Visualising Graphical covariance structures

Oracle panel
Plots from section 8: corr and stab along traj, corr decay, traj_comp, overlap
Biplots (need to integrate between utils and plot utils)

## Interacting with packages and import issues

File system:
- could add paths with sys and os
- copilot suggested could:
  Install src as a package: If src is a Python package (i.e., it contains an `__init__.py` file), you can install it with pip. Navigate to the parent directory of src in your terminal and run `pip install -e .`. This installs the current directory as a package in editable mode, which means you can modify the code in the package without having to reinstall it. After doing this, you should be able to import src from anywhere.
- for now will use relative imports (final copilot suggestion)

Changed mind and did a pip install -e. This required creating a setup file. Don't know if best practice, but seems to work, hopefully can consult with someone more experienced at some point and get some feedback on this.

