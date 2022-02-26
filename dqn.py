import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import actions
from actions import *
import pygame
from env import Player, World
import torch

SIZE = 84
NUM_ACTIONS = 5
MEMORY_CAPACITY = 30000
LR = 0.001
E = 0.01
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32
GAMMA = 0.90
EPISODES = 1000
FRAMES = 3000
SPEEDUP = 100
FRAME_SKIP = 4
STACK_SIZE = 4
STATE_SIZE = STACK_SIZE * SIZE ** 2


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


class DQN:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.eval_net, self.target_net = Net().to(self.device), Net().to(self.device)
        self.memory = torch.zeros((MEMORY_CAPACITY, STATE_SIZE * 2 + 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

    def choose_action(self, state):
        if np.random.randn() <= E:
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
        batch_state = batch_memory[:, : STATE_SIZE].reshape(-1, STACK_SIZE, SIZE, SIZE)
        batch_action = batch_memory[:, STATE_SIZE : STATE_SIZE + 1].type(torch.LongTensor).to(self.device)
        batch_reward = batch_memory[:, STATE_SIZE + 1 : STATE_SIZE + 2]
        batch_next_state = batch_memory[:, -STATE_SIZE:].reshape(-1, STACK_SIZE, SIZE, SIZE)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # [[q_1], [q_2], ..., [q_N]]
        q_next = self.target_net(batch_next_state).detach()          # [predictions1, predictions2, ...]
        q_target = batch_reward + GAMMA * q_next.max(1)[0].T.unsqueeze(1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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

    cpu_player = Player('Steve', 20, 20, 1)
    cpu_player.automate = True

    env.register_players([my_player])

    for i in range(EPISODES):
        env.reset()

        action = actions.NOTHING

        reward = 0
        episode_reward = 0
        reward_history = []
        score = env.get_score(my_player)

        image_buffer = torch.zeros(STACK_SIZE + 1, SIZE, SIZE)

        for frame_idx in range(FRAMES):
            time.sleep(0.016 / SPEEDUP)

            env.update()

            if frame_idx % FRAME_SKIP == 0:
                env.render(my_player.view, display_score=False)

                image = torch.Tensor(env.screenshot(my_player.view, SIZE, SIZE))
                image = (image - 128) / (128 - 1)

                for k in range(STACK_SIZE):
                    image_buffer[k] = image_buffer[k + 1]
                image_buffer[STACK_SIZE] = image

                dqn_input = torch.FloatTensor(image_buffer[-STACK_SIZE:])[None, :]

                # calculate reward
                reward = env.get_score(my_player) - score

                score = env.get_score(my_player)

                # if reward > 0:
                #     reward = 1
                # elif reward < 0:
                #     reward = -1
                # else:
                #     reward = 0

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


            env.perform_action(player=my_player, action=action)

            #cv2.imshow('CNN Input', image)

            #
            # keys = pygame.key.get_pressed()  # checking pressed keys
            # if keys[pygame.K_UP]:
            #     env.perform_action(my_player, MOVE_UP)
            # elif keys[pygame.K_DOWN]:
            #     env.perform_action(my_player, MOVE_DOWN)
            # elif keys[pygame.K_LEFT]:
            #     env.perform_action(my_player, MOVE_LEFT)
            # elif keys[pygame.K_RIGHT]:
            #     env.perform_action(my_player, MOVE_RIGHT)
            # else:
            #     env.perform_action(my_player, NOTHING)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # if event.type == pygame.KEYDOWN:
                #     if event.key == pygame.K_a:
                #         env.perform_action(my_player, SHOOT_LEFT)
                #     if event.key == pygame.K_d:
                #         env.perform_action(my_player, SHOOT_RIGHT)
                #     if event.key == pygame.K_w:
                #         env.perform_action(my_player, SHOOT_UP)
                #     if event.key == pygame.K_s:
                #         env.perform_action(my_player, SHOOT_DOWN)

        print("episode: {} , the episode reward is {}".format(i, round(episode_reward, 3)))

        ax.plot(reward_history, 'g-', label='total_loss')
        plt.show()