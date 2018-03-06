import numpy as np
import gym
import pdb
import math

from pprint import pprint

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam,RMSprop

from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
#from keras.models import model_from_json, Model
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.optimizers import Adam
#from keras.optimizers import RMSprop

from ReplayBuffer import ReplayBuffer
#from ActorNetwork import ActorNetwork
#from CriticNetwork import CriticNetwork
from OU import OU
import timeit
window_length=4
OU = OU()       #Ornstein-Uhlenbeck Process

ENV_NAME="torcs"

class MyProcessor(Processor):
    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        o=observation
        res=[o.speedX, o.speedY, o.speedZ, o.angle, o.rpm, o.trackPos]
        res.extend(o.track)
        res.extend(o.wheelSpinVel)
        #pdb.set_trace()
        return res

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tupel (observation, reward, done, reward) with with all elements after being processed.
        """
        print("OrigO:" + str(observation))
        o = observation
        spdX = o.speedX * 20
        spdY = o.speedY * 20
        spdZ = o.speedZ * 20
        spd = math.sqrt(spdX*spdX + spdY*spdY +spdZ*spdZ)
        reward = spd - 10*o.angle*o.angle

        if (max(o.track)<0):
            reward=reward-10

        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        pprint("O: " +str(observation))
        pprint("R: " +str(reward))
        pprint("I: " +str(info))
        return observation, reward, done, info

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 4  #Steering/Acceleration/Brake
    state_dim = 29  #of sensors input

    np.random.seed(1337)

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=False,gear_change=False)
    nb_actions = 3  # left, nothing , right, break

    # Next, we build a very simple model regardless of the dueling architecture
    # if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
    # Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
    model = Sequential()
#    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Flatten(input_shape=(window_length,29)))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000000, window_length=window_length)
    policy = BoltzmannQPolicy(tau=1.)
    processor=MyProcessor()
    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                enable_dueling_network=True, dueling_type='avg',
                target_model_update=1e-2, policy=policy,
                processor=processor)
    dqn.compile(RMSprop(lr=1e-3), metrics=['mae'])
    #dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME))

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)

if __name__ == "__main__":
    playGame()
