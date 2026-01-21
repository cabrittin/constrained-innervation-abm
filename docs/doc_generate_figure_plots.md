## Generate figure plots.
Use scripts/publication_plots.py to generate figure plots.

Relative paths to pertinent files are defined in configs/publication_config.ini You will need to adjust the all the paths under [main] to reflect where you installed the cebraindev_abm data directory. If any other relative paths have been changed, then the file must be adjusted accordingly. 

### Figure 1c
Bimodal reproducibility distributions of empirical age groups.
```
python scripts/publication_plots.py figure_1c
```
### Figure 1e
Number of spatial domains distributions of empirical age groups.
```
python scripts/publication_plots.py figure_1e
```
### Figure 1f
Mean spatial domain size distributions of empirical age groups.
```
python scripts/publication_plots.py figure_1f
```

