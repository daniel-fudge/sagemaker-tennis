# DRL applied to the Unity "Tennis" Environment on AWS SageMaker
This repo trains a Deep Reinforcement Learning (DRL) agent to solve the Unity ML-Agents "Tennis" environment on AWS SageMaker.   

Please see this [repo](https://github.com/daniel-fudge/reinforcement-learning-tennis) for a DRL implementation in a traditional environment.   

## Motivation
This repo was generated to support the building of a custom trading environment and DRL alogorithm for portfolio optimization in this [repo](https://github.com/daniel-fudge/DRL-Portfolio-Optimization-Custom).

## Project Details
### Tennis Environment
The [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 
environment provided by [Unity](https://unity3d.com/machine-learning/) has two agents that control rackets to bounce a 
ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the 
ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball 
in play.   
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each 
agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward 
(or away from) the net, and jumping.  

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

### Solving the Environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each 
agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.  

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Running locally
To set up your python environment to run the code in this repository, follow the instructions below. 

1. If not already installed, install the Anaconda Python distribution from [here](https://www.anaconda.com/distribution/). 

1. Ensure you have the "Build Tools for Visual Studio 2019" installed from this 
[site](https://visualstudio.microsoft.com/downloads/). This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) 
may also be very helpful.  This was confirmed to work in Windows 10 Home.  

1. Create (and activate) a new `tennis` environment with Python 3.6 and the OpenAI [gym](https://github.com/openai/gym).
If you need to delete the `tennis` environment type:  `conda env remove --name tennis`.  To deactivate the environment 
type: `conda deactivate`.

    ```bash
    conda update -n base -c defaults conda -y
    conda env create -f environment.yml
    activate tennis
    ```

1. Install CPU version of PyTorch 1.5.1. 

    ```bash
   conda install pytorch cpuonly -c pytorch -y
    ```

1. [OPTIONAL] Install GPU version of PyTorch 1.5.1 if you have a local GPU.

    ```bash
   conda install pytorch -c pytorch -y
    ```
