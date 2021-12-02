import numpy as np
import torch
import torch.optim as optim
import wandb
from gym import Wrapper
from gym_maze.envs.maze_env import *
import argparse

from config import config
from embedding_model import EmbeddingModel, compute_intrinsic_reward
from memory import Memory, LocalBuffer
from model import R2D2


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


class Maze(Wrapper):
    def step(self, action: int):
        obs, rew, done, info = super().step(["N", "E", "S", "W"][action])
        self.set.add((obs[0], obs[1]))
        if rew > 0:
            rew = 10
        return obs / 10, rew, done, info

    def reset(self):
        self.set = set()
        return super().reset()


def main(arglist):
    if arglist.env == "MazeEnvSample5x5":
        env = Maze(MazeEnvSample5x5())
    elif arglist.env == "MazeEnvSample10x10":
        env = Maze(MazeEnvSample10x10())
    elif arglist.env == "MazeEnvRandom5x5":
        env = Maze(MazeEnvRandom5x5())
    elif arglist.env == "MazeEnvRandom10x10":
        env = Maze(MazeEnvRandom10x10())
        
    config.enable_ngu = arglist.ngu
    
    torch.manual_seed(config.random_seed)
    env.seed(config.random_seed)
    np.random.seed(config.random_seed)
    env.action_space.seed(config.random_seed)

    wandb.init(project=arglist.project, config=config.__dict__)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
    update_target_model(online_net, target_net)
    embedding_model = EmbeddingModel(obs_size=num_inputs, num_outputs=num_actions)
    embedding_loss = 0

    optimizer = optim.Adam(online_net.parameters(), lr=config.lr)

    online_net.to(config.device)
    target_net.to(config.device)
    online_net.train()
    target_net.train()
    memory = Memory(config.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer()
    sum_reward = 0
    sum_augmented_reward = 0
    sum_obs_set = 0
    success_times = 0

    for episode in range(arglist.episode):
        done = False
        state = env.reset()
        state = torch.Tensor(state).to(config.device)

        hidden = (
            torch.Tensor().new_zeros(1, 1, config.hidden_size),
            torch.Tensor().new_zeros(1, 1, config.hidden_size),
        )

        episodic_memory = [embedding_model.embedding(state)]

        episode_steps = 0
        horizon = arglist.horizon
        while not done:
            steps += 1
            episode_steps += 1

            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            next_state, env_reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state)

            augmented_reward = env_reward
            if config.enable_ngu:
                next_state_emb = embedding_model.embedding(next_state)
                intrinsic_reward = compute_intrinsic_reward(episodic_memory, next_state_emb)
                episodic_memory.append(next_state_emb)
                beta = 0.0001
                augmented_reward = env_reward + beta * intrinsic_reward

            mask = 0 if done else 1

            local_buffer.push(state, next_state, action, augmented_reward, mask, hidden)
            hidden = new_hidden
            if len(local_buffer.memory) == config.local_mini_batch:
                batch, lengths = local_buffer.sample()
                td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)

            sum_reward += env_reward
            state = next_state
            sum_augmented_reward += augmented_reward
            if env_reward > 0:
                success_times += 1

            if steps > config.initial_exploration and len(memory) > config.batch_size:
                epsilon -= config.epsilon_decay
                epsilon = max(epsilon, 0.4)

                batch, indexes, lengths = memory.sample(config.batch_size)
                loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths)

                if config.enable_ngu:
                    embedding_loss = embedding_model.train_model(batch)

                memory.update_priority(indexes, td_error, lengths)

                if steps % config.update_target == 0:
                    update_target_model(online_net, target_net)

            if episode_steps >= horizon or done:
                sum_obs_set += len(env.set)
                break

        if episode > 0 and episode % config.log_interval == 0:
            mean_reward = sum_reward / config.log_interval
            mean_augmented_reward = sum_augmented_reward / config.log_interval
            mean_success_rate = success_times / config.log_interval
            metrics = {
                "episode": episode,
                "mean_reward": mean_reward,
                "epsilon": epsilon,
                "embedding_loss": embedding_loss,
                "loss": loss,
                "mean_augmented_reward": mean_augmented_reward,
                "steps": steps,
                "sum_obs_set": sum_obs_set / config.log_interval,
                "mean_success_rate": mean_success_rate
            }
            print(metrics)
            wandb.log(metrics)
            
            path = "./model/"+arglist.project+"-"+("ngu-" if arglist.ngu else "")
            torch.save(online_net, path + 'online_net.pth')
            torch.save(target_net, path + "target_net.pth")

            sum_reward = 0
            sum_augmented_reward = 0
            sum_obs_set = 0
            success_times = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",type=str,default="")
    parser.add_argument("--env",type=str,default="")
    parser.add_argument("--episode",type=int,default=1000)
    parser.add_argument("--ngu",action="store_true", default=False)
    parser.add_argument("--horizon",type=int,default=100)
    arglist = parser.parse_args()
    main(arglist)
