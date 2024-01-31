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

