# 2025 DomainIndependentNavigation
This projects shows how an independent Navigation algorithm can be trained and perform navigation tasks on a segmented Bird-Eye View Map.


## Train the Control Network
To Train the Control Network, the environment "CarRacing V3" from OpenAI Gymnasium is adapted.
In Combination with Stable-Baseline-3 a Network can be trained fast on a bird-eye-view segmented mask of the track ahead.

To start the training process run
```bash
python3 ControlNetworkTraining/train.py
```