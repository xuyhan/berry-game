import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import actions
from actions import *
import pygame
from env import Player, World
import torch

SIZE_W = 5  # 6
SIZE_H = 1  # 1
NUM_ACTIONS = 4
MEMORY_CAPACITY = 30000
LR = 0.001
E = 0.05
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32
GAMMA = 0.90
EPISODES = 10000
FRAMES = 3000
SPEEDUP = 100
FRAME_SKIP = 1
STACK_SIZE = 2
STATE_SIZE = STACK_SIZE * SIZE_H * SIZE_W


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(STACK_SIZE, 16, kernel_size=7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(256, NUM_ACTIONS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.fc1(x)

        return x


class NetSmall(nn.Module):
    def __init__(self):
        super(NetSmall, self).__init__()

        self.conv1 = nn.Conv2d(STACK_SIZE, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc1(x)

        return x


class NetBasic(nn.Module):
    def __init__(self):
        super(NetBasic, self).__init__()

        self.fc1 = nn.Linear(5 * STACK_SIZE, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, NUM_ACTIONS)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class DQN:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.eval_net, self.target_net = NetBasic().to(self.device), NetBasic().to(self.device)
        self.memory = torch.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # nn.CrossEntropyLoss()
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.steps = 0

    def choose_action(self, state):
        # E_ = .5 if self.steps < 100 else 1 / (5 * np.log(self.steps))
        # E_ = 1 / (2 * np.log(0.001 * self.steps + 1.8))
        E_ = max(0.02, - 1 / 20000 * self.steps + 0.9)
        k = np.random.rand()

        if k >= E_:
            state = state.to(self.device)
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(0, NUM_ACTIONS)

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = torch.hstack((state.flatten(), torch.FloatTensor([action, reward]), next_state.flatten()))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition.to(self.device)
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_counter, MEMORY_CAPACITY), BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = batch_memory[:, : STATE_SIZE].reshape(-1, STACK_SIZE, SIZE_H, SIZE_W)
        batch_action = batch_memory[:, STATE_SIZE: STATE_SIZE + 1].type(torch.LongTensor).to(self.device)
        batch_reward = batch_memory[:, STATE_SIZE + 1: STATE_SIZE + 2]
        batch_next_state = batch_memory[:, -STATE_SIZE:].reshape(-1, STACK_SIZE, SIZE_H, SIZE_W)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # [[q_1], [q_2], ..., [q_N]]
        q_next = self.target_net(batch_next_state).detach()  # [predictions1, predictions2, ...]
        q_target = batch_reward + GAMMA * q_next.max(1)[0].T.unsqueeze(1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1


if __name__ == '__main__':
    import time, cv2
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    dqn = DQN()

    pygame.init()
    pygame.font.init()

    running = True

    env = World(10, 10)

    my_player = Player('Alex', 0, 0, 2)
    my_player.automate = False

    # cpu_player = Player('Steve', 20, 20, 1)
    # cpu_player.automate = True

    env.register_players([my_player])
    speedup = 100
    user_mode = False
    input_cd = 0

    for i in range(EPISODES):
        env.reset()

        action = actions.NOTHING

        reward = 0
        episode_reward = 0
        reward_history = []
        score = env.get_score(my_player)

        image_buffer = torch.zeros(STACK_SIZE + 1, SIZE_H, SIZE_W)

        env.render(my_player.view, display_score=False)

        for frame_idx in range(FRAMES):
            if speedup < 10 and not user_mode:
                time.sleep(1 / speedup)
            elif user_mode:
                time.sleep(1 / 60)

            env.update()

            if frame_idx % FRAME_SKIP == 0:
                # image = torch.Tensor(env.screenshot(my_player.view, SIZE_W, SIZE_H))
                # image = (image - 128) / (128 - 1)

                # image = torch.Tensor(env.screenshot_basic(my_player.view))

                image = torch.Tensor(env.screenshot_positions())

                for k in range(STACK_SIZE):
                    image_buffer[k] = image_buffer[k + 1]
                image_buffer[STACK_SIZE] = image

                dqn_input = torch.FloatTensor(image_buffer[-STACK_SIZE:])[None, :]

                # calculate reward
                if not env.kill:
                    reward = env.get_score(my_player) - score

                    if reward > 0:
                        reward = 1
                    elif reward < 0:
                        reward = -1
                else:
                    reward = -1

                score = env.get_score(my_player)

                episode_reward += reward
                reward_history.append(episode_reward)

                # store transition from previous state to current state
                s_last = torch.FloatTensor(image_buffer[:STACK_SIZE])
                s_current = torch.FloatTensor(image_buffer[-STACK_SIZE:])
                dqn.store_transition(s_last, action, reward, s_current)

                # next action
                action = dqn.choose_action(dqn_input)

                if dqn.memory_counter >= BATCH_SIZE * 3:
                    dqn.learn()

                if env.kill:
                    break

            if not user_mode:
                env.perform_action(player=my_player, action=action + 1)

            keys = pygame.key.get_pressed()  # checking pressed keys
            if keys[pygame.K_e]:
                speedup = 100
            elif keys[pygame.K_q]:
                speedup = 3

            input_cd = max(0, input_cd - 1)

            if user_mode and input_cd == 0:
                if keys[pygame.K_UP]:
                    env.perform_action(my_player, MOVE_UP);
                    input_cd = 10
                elif keys[pygame.K_DOWN]:
                    env.perform_action(my_player, MOVE_DOWN);
                    input_cd = 10
                elif keys[pygame.K_LEFT]:
                    env.perform_action(my_player, MOVE_LEFT);
                    input_cd = 10
                elif keys[pygame.K_RIGHT]:
                    env.perform_action(my_player, MOVE_RIGHT);
                    input_cd = 10
                elif keys[pygame.K_SPACE]:
                    env.perform_action(my_player, EAT);
                    input_cd = 10
                else:
                    env.perform_action(my_player, NOTHING)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            env.render(my_player.view, display_score=False)

        print("episode: {} , the episode reward is {}, score is {}".format(i, round(episode_reward, 3),
                                                                           round(score, 3)))

        ax.plot(reward_history, 'g-', label='total_loss')
        plt.show()
