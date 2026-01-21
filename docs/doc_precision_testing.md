# Precision testing
This page demonstrates how precison can modified in the simulation. Precision determines how frequently a growth step will be towards a pioneer target. Precision is adjusted using the "agent_precision" field in the config (.ini) file. Precison is a float between 0 and 1, where 1 indicates perfect precision. 

In practice, at every growth step and for each agent, random number r is drawn. If r is less than the precision, then the agent will respond the the pioneer attractor field. If r is greater than the precision, then the agent will grow as if there is no attractor field. 

Below demonstrates the effects of changing precision for a model with 6 agents and 2 pioneers. Pioneer agents are red. Follower agents with the same color are assigned to the same pioneer. 

### Precision: 1.0
```
python test/viz_test_model.py viz_simulation configs/model_precision_high.ini
```
![High precision](./images/precision_high_1.gif)  

### Precision: 0.5
```
python test/viz_test_model.py viz_simulation configs/model_precision_med.ini
```
![Medium precision](./images/precision_med_1.gif)  

### Precision: 0.1
```
python test/viz_test_model.py viz_simulation configs/model_precision_low.ini
```
![Low precision](./images/precision_low_1.gif) 

 
