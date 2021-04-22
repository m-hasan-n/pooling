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

## NGSIM Dataset Pre-processing

