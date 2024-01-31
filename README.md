What are the different types of plot that need to be fitted into the new framework?

Data Histogram
Bubble plots and derivatives
Visualising Graphical covariance structures

Oracle panel
Plots from section 8: corr and stab along traj, corr decay, traj_comp, overlap
Biplots (need to integrate between utils and plot utils)



Tests:
File structure doesn't matter too much I think.
Best first to get some decent tests down then can rearrange as desired.

File system:
- could add paths with sys and os
- copilot suggested could:
  Install src as a package: If src is a Python package (i.e., it contains an `__init__.py` file), you can install it with pip. Navigate to the parent directory of src in your terminal and run `pip install -e .`. This installs the current directory as a package in editable mode, which means you can modify the code in the package without having to reinstall it. After doing this, you should be able to import src from anywhere.
- for now will use relative imports (final copilot suggestion)


Next tasks:
- Nutrimouse pbootstrap fit and oracle panel recover
    See how the old code base did the saving and loading for parametric bootstrap and port that over

Later tasks:
- Add a test that creates synthetic covariance and check that CCA function recovers desired structure
- 