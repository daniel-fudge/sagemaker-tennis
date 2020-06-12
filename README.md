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
