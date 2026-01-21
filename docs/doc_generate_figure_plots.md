## Generate figure plots.
Use scripts/publication_plots.py to generate figure plots.

### Setting up file paths in publication\_contig.ini

Relative paths to pertinent files are defined in configs/publication\_config.ini. You will need to adjust the all the paths under [main] to reflect where you installed data directory. If any other relative paths have been changed, then the file must be adjusted accordingly. Below are the entries in the publication\_config.ini that need to be modified:
```
[main]
# Path to the data directory where you have the simulations 'models' directory
data=./data 
# Path to the empirical data
data_emp=${data}/empirical
# Path to where figures will be saved
results=./results/pub_figs
# Toggle True/False to show plots when running script
show_plot=True
# Path to the directory with the config (.ini) files
cfg_dir=./configs
# Path to single-cell data
data_sc=./data/single_cell/rosette_paper
```

### Making the figure plot

To generate a figure from the paper simply specify the figure and panel. For example, the following will generate Figure 5E:
```
python scripts/publication_plots.py figure_1e
```
Note: additional visual editing may have been performed in a graphics editor (e.g. Inkscape). 

The following will generate Figure S6A
```
python scripts/publication_plots.py figure_s6a
```
Note that the 's' denotes supplemental.

