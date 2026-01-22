## Format data intermediates  

All data intermediates are downloadable from the Zenodo link associated
with the paper. These intermediates consist of processed files derived
from the raw data and are used for anaylsis and figure generation.

Should you want to generate the intermediates yourself, then you can used
the script/format\_data.py script. You will need to modify the paths in
the configs/format\_data.ini. Below are the paths that will need to be
modified based on where the data is on your system: 
``` 
[main] 
#Path to directory where the data is stored 
data=./data/ 
#Path to directory where results will outputted 
results=./results 
#Path to directory where supplemental data is kept
supp_data=${results}/supp_data 
#Path to directory where empirical data is kept 
data_emp=${data}/empirical 
#Path to directory where simulated models are kept 
data_sim=${data}/models 
# Path to the directory with the config (.ini) files 
cfg_dir=./configs 
# Path to single-cell data 
data_sc=${home}/data/single_cell/rosette_paper 
```

To format intermediates simply pass the sections names in the ini file
(section names are in brackets \[...\]) to the script e.g., 
``` 
python scripts/format_data.py empirical_degree_dist 
```

If you need an intermeidate for figure generation, you can cross reference
the publication\_plots.ini and format\_data.ini to determine the
appropriate section name. 



