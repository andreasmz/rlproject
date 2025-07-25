# 2048 Reinforcement Learning

This repository contains the code for Reinforcement Learning on the game 2048 for the course 'Reinforcement Learning' at Johannes Gutenberg University Mainz (summer term 2025). The techniques used are
* Game environment implementation in Gymnasium using numpy
* N-Tuple networks with customly designed 4-tuples
* TD(0), TD(lambda) and MC learning
* Replay buffer to stabilize training

Also, experiments were performed for other techniques:
* Neuronal networks for weighted voting of the tuples
* Deep neuronal networks
* Convolution
* DQN

### Watch the agent reaching the 2048 tile

![Agent reaches tile 2048](media/n4-tuple_network_MC_avg_vote/n4_avg_MC_lambda_2048_score_27172_02.gif)

Here you can find the full video

[![Agent playing 2048](media/n4-tuple_network_MC_avg_vote/n4_avg_MC_lambda_2048_score_27172.mp4)](media/n4-tuple_network_MC_avg_vote/n4_avg_MC_lambda_2048_score_27172.mp4)


### Installation

The project has three parts:
* The numpy / gymnasium implementation as a standalone module
* The final notebooks + trained model found in the project root
* The dev folder containing the notebooks with experiments

The module is designed to also contain the dependencies for the notebooks. To install, simply tipe

```bash
conda create -n andreas2024 python=3.13
pip install 
```