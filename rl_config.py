import random
import sys
class RLConfig:
    def __init__(self):
        self.title="No Name"
        self.learning_rate=0.0001
        self.useDoubleDQN=True
        self.useDuelingDQN=True
        self.usePER=True
        self.eps_start=1.0
        self.eps_min=0.01
        self.eps_decay=0.9996
        self.num_episodes_per_evaluation=500
        self.num_evaluations=3
        self.print_interval=100
        self.beta0=0.1
        self.beta_annealing_timesteps=100000
        self.betaf=1.0
        self.gamma=0.999
        self.tau=0.001
        self.per_alpha=0.2
        self.per_epsilon=5e-4
        self.samples_before_learning=1000
        self.buffer_size=int(1e5)
        self.batch_size=64
        self.parameter_update_interval=4
        self.learning_interval=1
        self.verbose=False
        self.modelSaveName="dummy.pth"
        self.seed=random.randint(0, sys.maxsize)
