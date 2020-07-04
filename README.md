# DRL applied to the Unity "Tennis" Environment on AWS SageMaker
This repo trains a Deep Reinforcement Learning (DRL) agent to solve the Unity ML-Agents "Tennis" environment on AWS SageMaker.   

Please see this [repo](https://github.com/daniel-fudge/reinforcement-learning-tennis) for a DRL implementation in a traditional environment.   

## Motivation
This repo was generated to support the building of a custom trading environment and DRL algorithm for portfolio optimization in this [repo](https://github.com/daniel-fudge/DRL-Portfolio-Optimization-Custom).

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

## Running locally
### First Setup and execution
To set up your python environment to run the code in this repository, follow the instructions below. 

**_Note:_** These commands assume you are in a Windows PowerShell terminal.  If using an traditional command prompt you 
may have to slightly alter some of the commands. 
1. Unpack the Windows Tennis executable into the `src` folder.  Note if corrupted you can get fresh version [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).

   ```cmd
   expand-archive container\Tennis_Windows_x86_64.zip container\src
   ```

1. If not already installed, install the Anaconda Python distribution from [here](https://www.anaconda.com/distribution/). 

1. Ensure you have the "Build Tools for Visual Studio 2019" installed from this 
[site](https://visualstudio.microsoft.com/downloads/). This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) 
may also be very helpful.  This was confirmed to work in Windows 10 Home.  

1. Create (and activate) a new `tennis` environment with Python 3.6 and the OpenAI [gym](https://github.com/openai/gym).
If you need to delete the `tennis` environment type:  `conda env remove --name tennis`.  To deactivate the environment 
type: `conda deactivate`.

    ```cmd
    conda update -n base -c defaults conda -y
    conda env create -f environment.yml
    conda activate tennis
    ```

1. Install CPU version of PyTorch 1.1.0 (version matched to SageMaker). 

    ```cmd
   conda install pytorch=1.1.0 cpuonly -c pytorch -y
    ```

1. [OPTIONAL] Install GPU version of PyTorch 1.1.0 if you have a local GPU.

    ```cmd
    conda install pytorch=1.1.0 -c pytorch -y
    ```

1. Execute the training process simply double click the `train.bat` file.  You should get an average score of 0.3 after 
1700 episodes.  A score of 0.5 solves the environment.

    ```cmd
    train.bat
    ```
   
### Subsequent execution
Assuming you have completed the above steps you can skip directly to the execution.    

```cmd
conda activate tennis
train.bat
```

## Running on AWS Sagemaker
Please see [this](https://youtu.be/w2r8ffcBVSo) video if you are unfamiliar with starting an AWS Sagemaker Notebook 
instance. 

Running on AWS is divided into the following four Notebooks.
1. [Building and Testing CPU Image](build-cpu.ipynb)   
    - Describes the Docker image creation process
    - Builds the CPU Docker image and publishes it to [ECR](https://aws.amazon.com/ecr/)
    - Tests the image locally on the Notebook instance, useful for debugging
    - Spawns a remote training job with the CPU image
    - Retrieves the results from the associated [S3](https://aws.amazon.com/s3/) bucket 
1. [Building and Testing GPU Image](build-gpu.ipynb)
    - Builds the GPU Docker image and publishes it to [ECR](https://aws.amazon.com/ecr/)
    - Spawns a remote training job with the GPU image
    - Retrieves the results from the associated [S3](https://aws.amazon.com/s3/) bucket 
1. [Cost and Timing Sensitivity](sensitivity.ipynb)
    - Compares the training [cost](https://aws.amazon.com/sagemaker/pricing/) and times on difference instance [types](https://aws.amazon.com/sagemaker/pricing/instance-types/)
1. [Hyper-parameter Tuning](tuning.ipynb)
    - After determining the most cost and time effective instance type, tune the hyper-parameters
    - This determines the set of hyper-parameters that minimizes the number of training epochs
