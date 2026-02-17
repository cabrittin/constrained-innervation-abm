## Run models
This page describes how to run simulations.

## Directory structure
We recommend that you setup a directory for each new model that you run. These instruction assume that your model will be setup in the directory data/models/model1. When you setup your directory, you will need to create a model.ini config file which specifies the model parameters. Two template config files are included — configs/model\_sweep.ini and configs/model\_sample.ini — which you can copy over depending on your needs. When copying over the template make sure to name model.ini.   

## The config file: model.ini
The choice of template file depends on whether you are doing a basic parameter sweep (model\_sweep.ini) or if you randomly sampling parameters to peturb (model\_sample.ini) like in the robustness experiments (Fig. 6). Details of the config files and be found here: [model\_sweep.ini](doc_model_sweep.md) and [model\_sample.ini](doc_model_sample.md). 
