# Maneuver-Aware Pooling for Vehicle Trajectory Prediction
Predicting the behavior of 
surrounding human drivers is vital for autonomous vehicles to share 
the same road with humans. Behavior of each of the surrounding 
vehicles is governed by the motion of its neighbor vehicles. 
This project focuses on predicting the behavior of the surrounding 
vehicles of an autonomous vehicle on highways. 
We are motivated by improving the prediction accuracy when a 
surrounding vehicle performs lane change and highway merging 
maneuvers. We propose a novel pooling strategy to capture 
the inter-dependencies between the neighbor vehicles. 
Depending solely on Euclidean trajectory representation, 
the existing pooling strategies do not model the context 
information of the maneuvers intended by a surrounding vehicle. 
In contrast, our pooling mechanism employs polar trajectory 
representation, vehicles orientation and radial velocity. 
This results in an implicitly maneuver-aware pooling operation.
We incorporated the proposed pooling mechanism into a generative
encoder-decoder model, and evaluated our method on the public 
NGSIM dataset.

#
![model image](pooling_model.png "Model overview")

## Pooling Toolbox
This project helps to reproduce 
the proposed and other pooling approaches such as Social LSTM , Covolutional Social Pooling and Soicla GAN.
#
![pooling image](pooling_approaches.png "Pooling approaches")

Visualizing pooling mechanisms (A green vehicle shows the ego, 
yellow vehicle shows a neighbor covered by the pooling strategy,
and grey vehicle shows a non-covered neighbor). 
* Left: a spatial grid is centered around the ego vehicle. 
The social tensor is structured accordingly and populated
with LSTM states of the ego and exisiting neighbor vehicles. 
  The social tensor is used with Social LSTM and Covolutional Social Pooling works.
#  
* Center: relative positions between the ego vehicle and 
  all its neighbors are concatenated to vehicle LSTM states. This is 
  the pooling strategy used in Social GAN work.
#  
* Right: the proposed pooling strategy where vehicle LSTM 
  states are concatenated to relative polar positions 
  (distance and angle) rather than the Cartesian representation
  used by the previous works.
  
## NGSIM Dataset Pre-processing

