## Run models
This page describes how to run simulations.

## Directory structure
We recommend that you setup a directory for each new model that you run. These instruction assume that your model will be setup in the directory data/models/model1. When you setup your directory, you will need to create a model.ini config file which specifies the model parameters. Two template config files are included — configs/model\_sweep.ini and configs/model\_sample.ini — which you can copy over depending on your needs. When copying over the template make sure to name model.ini.   

## The config file: model.ini
The choice of template file depends on whether you are doing a basic parameter sweep (model\_sweep.ini) or if you randomly sampling parameters to peturb (model\_sample.ini) like in the robustness experiments (Fig. 6). Details of the config files and be found here: [model\_sweep.ini](doc_model_sweep.md) and [model\_sample.ini](doc_model_sample.md).

## Build `sweep_log` file
The `sweep_log` file specifies the parameters for each simulations. Each row is a set of 6 simulation replicates. If doing the basic sweep:
```
> python scripts/run_model.py build_sweep --dir data/models/model1/     
Combos:: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 70/70 [00:00<00:00, 7380.25it/s]
Wrote to data/models/model1/sweep_log.csv

```
If doing the sampler:
```
> python scripts/run_model.py build_sampler --dir data/models/model1/ 
Samples: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 123969.58it/s]
Wrote to data/models/model1/sweep_log.csv
```

## Run the model
To run the model:
```
> python scripts/run_model.py run --dir data/models/model1/
Data written to data/models/model1/sims/
  0%|▏                            | 4/7000 [00:08<3:53:54,  2.01s/it]
```

The ouput will be .pkl files in the `data/models/model1/sims` directory. One .pkl file per line in the `sweep_log` file. 

## Analyze the simulations
To analyze the simulations:
```
> python scripts/run_model.py run --dir data/models/model1/ --fout data/models/model1/dataframe.csv
Data written to data/models/model1/sims/
  0%|▏                            | 1/7000 [00:14<27:45:20, 14.28s/it]]
```
Analysis takes a long time because of the spatial domain clustering. The output is the `dataframe.csv` where each row corresponds to the analysis output to the corresponding row in the `sweep_log` file.  

## Parallelizing simulation runs and analysis
Simulations are already parallized to use 6 processing units, where each unit simulates a replicate for the specified parameters. If your machine has at least 16 processing units, then additional parallelization is possible. For linux machines, we provide bash scripts to facilitate parallelization. These can be executed as follows. To paralellize runs:
```
>./bash/parallelize_run.sh 2 data/models/model1/
Parallelize data/models/model_test/ across 2 procs
Log split: 0:35 35:70
Data written to data/models/model_test/sims/
Data written to data/models/model_test/sims/
Done!        | 2/3500 [00:03<1:39:41,  1.71s/it]
```
Here `2` splits the `sweep_log` over two processing units, but keep in mind that each processing unit will spin up an additonal 6 processing units. 

To parallelize the analysis:
```
> ./bash/parallelize_analyze.sh 2 data/models/model1/
Parallelize across 2
Saving to data/models/model_test//dataframe.csv
Log split: 0:35 35:70
Initiating analysis for tmp/split.tmp.2
Initiating analysis for tmp/split.tmp.1
```

To run both in sequentially: 
```
> ./bash/parallel_run_and_analyze.sh 2 data/models/model1
```






