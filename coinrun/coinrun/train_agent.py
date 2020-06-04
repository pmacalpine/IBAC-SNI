"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
from mpi4py import MPI
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config

#from baselines.ppo2 import ppo2
#from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus


from procgen import ProcgenEnv

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)

import sys

def main():
    args = setup_utils.setup_and_load()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    seed = int(time.time()) % 10000
    set_global_seeds(seed * 100 + rank)

    #utils.setup_mpi_gpus()
    setup_mpi_gpus()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    nenvs = Config.NUM_ENVS
    
    total_timesteps = int(160e6)
    if Config.LONG_TRAINING:
        total_timesteps = int(200e6)
    elif Config.SHORT_TRAINING:
        total_timesteps = int(120e6)
    save_interval = args.save_interval

    total_timesteps = int(25e6)
    #env = utils.make_general_env(nenvs, seed=rank)
    #print (env)

    print (Config.ENVIRONMENT)
    
    print ("Rank")
    print (rank)
    print ("num_levels")
    print (Config.NUM_LEVELS)
    venv = ProcgenEnv(num_envs=nenvs, env_name=Config.ENVIRONMENT, num_levels=Config.NUM_LEVELS, paint_vel_info=Config.PAINT_VEL_INFO, distribution_mode="easy")
    #print (venv)

    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)
    
    #sys.exit(0)
    with tf.Session(config=config) as sess:
        #env = wrappers.add_final_wrappers(env)
        venv = wrappers.add_final_wrappers(venv)
        
        policy = policies.get_policy()

        #sess.run(tf.global_variables_initializer())
        ppo2.learn(policy=policy,
                    env=venv,
                    #env=env,
                    save_interval=save_interval,
                    nsteps=Config.NUM_STEPS,
                    nminibatches=Config.NUM_MINIBATCHES,
                    lam=0.95,
                    gamma=Config.GAMMA,
                    noptepochs=Config.PPO_EPOCHS,
                    log_interval=1,
                    ent_coef=Config.ENTROPY_COEFF,
                    lr=lambda f : f * Config.LEARNING_RATE,
                    cliprange=lambda f : f * 0.2,
                    total_timesteps=total_timesteps)

if __name__ == '__main__':
    main()

