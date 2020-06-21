"""
This script trains and saves the model and plots its performance.

Note:  You need to verify the env path is correct for you PC and OS.
"""

from tennis.train import make_plot, setup, train
import os
import platform
from time import time
import torch
from unityagents import UnityEnvironment

# !!!!!!!!! YOU MAY NEED TO EDIT THIS !!!!!!!!!!!!!!!
if platform.system() == 'Windows':
    print("Loading Windows x86 64-bit Tennis environment.")
    env = UnityEnvironment(file_name=r"Tennis_Windows_x86_64\Tennis.exe")
elif platform.system() == 'Linux':
    print("Loading Linux Tennis environment.")
    env = UnityEnvironment(file_name=r"Tennis_Linux_NoVis/Tennis.x86_64")
else:
    print("Only Windows and Linux supported.")
    raise RuntimeError

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU activated.")
else:
    device = torch.device("cpu")
    print("CPU activated.")

if __name__ == "__main__":

    # Delete the old output files
    # -----------------------------------------------------------------------------------
    for name in ['score.png', 'scores.npz', 'checkpoint.pth']:
        if os.path.isfile(name):
            os.remove(name)

    # Setup the environment and agent
    # -----------------------------------------------------------------------------------
    agents = setup(env)

    # Perform the training
    # -----------------------------------------------------------------------------------
    print('Training the agent.')
    start = time()
    train(agents=agents, env=env)
    print("Training Time:  {:.1f} minutes".format((time() - start)/60.0))

    # Make some pretty plots
    # -----------------------------------------------------------------------------------
    print('Make training plot called rewards.png.')
    make_plot()
