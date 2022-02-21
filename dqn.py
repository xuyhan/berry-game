import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from actions import *
import pygame
from env import Player, World
import torch

SIZE = 64
NUM_ACTIONS = 4
MEMORY_CAPACITY = 1000
LR = 0.001
E = 0.05
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 128
GAMMA = 0.90
EPISODES = 100
FRAMES = 3600
SPEEDUP = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, NUM_ACTIONS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.fc1(x.flatten())

        return x


class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = torch.zeros((MEMORY_CAPACITY, (SIZE ** 2) * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0

    def choose_action(self, state):
        if np.random.randn() <= E:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, NUM_ACTIONS)

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, : SIZE ** 2])
        batch_action = torch.LongTensor(batch_memory[:, SIZE ** 2 : SIZE ** 2 + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, SIZE ** 2 + 1 : SIZE ** 2 + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, - (SIZE ** 2):])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    import time, cv2

    dqn = DQN()

    pygame.init()
    pygame.font.init()

    running = True

    env = World(20, 20)

    my_player = Player('Alex', 0, 0, 2)
    my_player.automate = True

    cpu_player = Player('Steve', 20, 20, 1)
    cpu_player.automate = True

    env.register_players([my_player, cpu_player])

    for i in range(EPISODES):
        env.reset()

        for _ in range(FRAMES):
            time.sleep(0.016 / SPEEDUP)  # Make the game slow down

            env.update()
            env.render(my_player.view, display_score=True)

            image = env.screenshot(my_player.view, SIZE, SIZE)

            action = dqn.choose_action(image[None, :])
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            #reward = reward_func(env, x, x_dot, thet


            #cv2.imshow('CNN Input', image)

            #image = (image - 128) / (128 - 1)

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
            #
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

