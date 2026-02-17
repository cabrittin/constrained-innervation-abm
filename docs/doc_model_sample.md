# Model Sample Configuration

## Overview

This configuration file is used to sample random perturbations to test the effects of lowering follower specificity and pioneer uniqueness. The file has the same basic setup as [model\_sweep.ini](doc_model_sweep.md) with the addition of the `sampler` key in the the `[sweep]` section. For the manuscript, only one sampler is used: `sampler_specficity`.

## Sweep file formats
When building the sweep files, the output formats for each sampler are described below

### sampler_specificity
```
Rows:
------
0: High specificity, low pioneer-follower distance  (control)
1: High specificity, variable pioneer-follower distance
2: 90% specificity, variable pioneer-follower distance
3: 80% specificity, variable pioneer-follower distance
4: 90% pioneer distinctiveness, variable pioneer-pioneer distance
5: 80% pioneer distinctiveness, variable pioneer-pioneer distance

These rows are repeated per parameter group

```
