[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Follow the instructions for constructing the udacity virtual environment found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). 
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in this repository.

### Instructions

Activate the drlnd environment, then launch the `Navigation.ipynb` jupyter notebook from a jupyter session. Follow the instructions there to train the agent. 

### Files and Descriptions

`Navigation.ipynb` directs the agent training and the saving of results.  
`dqn_agent.py` provides a general DQN agent and supports a few different architectural variants.  
`model.py` provides a pytorch neural network that will learn this environment.  
`rl_config.py` provides default hyperparameters and other configuration information.  
`segment_tree.py` from the [OpenAI baseline](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py). A tree structure used for prioritized experience replay licensed under the MIT license.  
`replay_buffer.py` from the [OpenAI baseline](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py). Provides the prioritized experience replay buffer as well the standard fixed-length replay buffer licensed under the MIT license.  
`dqn.pth` saved model weights for the baseline DQN architecture.  
`ddqn.pth` saved model weights for the double DQN architecture.  
`ddqn_per.pth` saved model weights for the Double DQN architecture with Prioritized Experience Replay.  
`dueling_dqn.pth` saved model weights for the Dueling DQN architecture.  
`dueling_dqn_per.pth` saved model weights for the Dueling DQN architecture with Prioritized Experience Replay.  
`dqn.pkl` training history from the DQN architecture.  
`ddqn.pkl` training history from the Double DQN architecture.  
`ddqn_per.pkl` training history from the Double DQN + PER architecture.  
`dueling_dqn.pkl` training history from the Dueling DQN architecture.  
`dueling_dqn_per.pkl` training history from the Dueling DQN + PER architecture.  
`Report.ipynb` final project report jupyter notebook.  
`Report.pdf` final project report in pdf format.

### External Sources
A clean room Prioritized Experience Replay implementation was deemed excessive for the purposes of this project. Instead, I have used the MIT licensed OpenAI baseline to provide this service as noted above. The same source that provides the PER buffer also provides the bog standard replay buffer so that was used although a clean room implementation of that is straightforward.