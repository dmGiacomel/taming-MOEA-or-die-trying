import gymnasium as gym 
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import os

models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make("LunarLander-v3")
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIME_STEPS = 10000
for i in range (1,30):
    model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False, tb_log_name="A2C")
    model.save(f"models_dir/{TIME_STEPS*i}")
env.close()
# env = gym.make("LunarLander-v3", render_mode="human")

# episodes = 10
# for ep in range(episodes):
#     obs, info = env.reset()
#     done = False

#     while not done:
#         env.render()
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)    
#         done = terminated or truncated
    
env.close()