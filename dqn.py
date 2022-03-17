import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import actions
from actions import *
import pygame
from env import Player, World
import torch
import random
from collections import defaultdict

SIZE_W = 84  # 6
SIZE_H = 84  # 1
NUM_ACTIONS = 7
BUFFER_SIZE = 29000
LR = 0.001
E = 0.05
TARGET_GAP = 100
BATCH_SIZE = 32
GAMMA = 0.9
EPISODES = 200
FRAMES = 400
SPEEDUP = 100
FRAME_SKIP = 1
STACK_SIZE = 1
N_PLAYERS = 2
STATE_SIZE = N_PLAYERS * STACK_SIZE * SIZE_H * SIZE_W
N_RANDOM_STATES = 100


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(STACK_SIZE * N_PLAYERS, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(256, NUM_ACTIONS)

        # self.conv1 = nn.Conv2d(STACK_SIZE, 64, kernel_size=8, stride=8)
        # self.pool = nn.MaxPool2d(3, 3)
        # self.fc1 = nn.Linear(576, NUM_ACTIONS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = self.fc1(x)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = x.flatten(1)
        # x = self.fc1(x)
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
    '''
    Adapted from: https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch
    '''

    def __init__(self, load=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.eval_net, self.target_net = Net().to(self.device), Net().to(self.device)

        self.train_ = True
        if load:
            self.eval_net.load_state_dict(torch.load(load))
            self.train_ = False

        self.memory = torch.zeros((BUFFER_SIZE, STATE_SIZE * 2 + 2)).to(self.device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()  # nn.CrossEntropyLoss()
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.steps = 0
        self.double_dql = True

    def choose_action(self, state):
        # E_ = .5 if self.steps < 100 else 1 / (5 * np.log(self.steps))
        # E_ = 1 / (2 * np.log(0.001 * self.steps + 1.8))
        E_ = max(0.001, - 1 / 20000 * self.steps + 0.9)
        # print(f'Epsilon: {E_}')
        k = np.random.rand()

        if not self.train_:
            E_ = 0

        if k >= E_:
            state = state.to(self.device)
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(0, NUM_ACTIONS)

        return action

    def store_transition(self, state, action, reward, next_state):
        transition = torch.hstack((state.flatten(), torch.FloatTensor([action, reward]), next_state.flatten()))
        index = self.memory_counter % BUFFER_SIZE
        self.memory[index, :] = transition.to(self.device)
        self.memory_counter += 1

    def evaluate(self):
        qs = 0
        count = min(self.memory_counter, BUFFER_SIZE)

        self.eval_net.eval()

        for i in range(count // BATCH_SIZE):
            batch_memory = self.memory[i * BATCH_SIZE : (i + 1) * BATCH_SIZE, :]
            batch_state = batch_memory[:, : STATE_SIZE].reshape(-1, STACK_SIZE * N_PLAYERS, SIZE_H, SIZE_W)
            qs += self.eval_net(batch_state).max(axis=1)[0].sum().item()

        if count % BATCH_SIZE > 0:
            batch_memory = self.memory[(count // BATCH_SIZE) * BATCH_SIZE : count, :]
            batch_state = batch_memory[:, : STATE_SIZE].reshape(-1, STACK_SIZE * N_PLAYERS, SIZE_H, SIZE_W)
            qs += self.eval_net(batch_state).max(axis=1)[0].sum().item()

        return qs / count

    def learn(self):
        if not self.train_:
            return

        if self.learn_step_counter % TARGET_GAP == 0 and not self.double_dql:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_counter, BUFFER_SIZE), BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = batch_memory[:, : STATE_SIZE].reshape(-1, STACK_SIZE * N_PLAYERS, SIZE_H, SIZE_W)
        batch_action = batch_memory[:, STATE_SIZE: STATE_SIZE + 1].type(torch.LongTensor).to(self.device)
        batch_reward = batch_memory[:, STATE_SIZE + 1: STATE_SIZE + 2]
        batch_next_state = batch_memory[:, -STATE_SIZE:].reshape(-1, STACK_SIZE * N_PLAYERS, SIZE_H, SIZE_W)

        self.eval_net.train()

        if not self.double_dql:
            q_eval = self.eval_net(batch_state).gather(1, batch_action)  # [[q_1], [q_2], ..., [q_N]]
            q_next = self.target_net(batch_next_state).detach()  # [predictions1, predictions2, ...]
            q_target = batch_reward + GAMMA * q_next.max(1)[0].T.unsqueeze(1)
            loss = self.loss_func(q_eval, q_target)
        else:
            q_eval = self.eval_net(batch_state).gather(1, batch_action)
            q_max = self.eval_net(batch_next_state).detach().max(1)[1]
            temp = self.target_net(batch_next_state).detach().gather(1, q_max.unsqueeze(1))
            q_target = batch_reward + GAMMA * temp
            loss = self.loss_func(q_eval, q_target)

            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1


if __name__ == '__main__':
    import time, cv2
    import matplotlib.pyplot as plt
    from env import LEFT, RIGHT

    fig, ax = plt.subplots()

    dqn = DQN()
    dqn_ = DQN()
    #dqn.train_ = False
    #print('#NUM PARAMETERS')
    #print(sum(p.numel() for p in dqn.eval_net.parameters()))

    pygame.init()
    pygame.font.init()

    running = True

    env = World(9, 9, n_players=2, start_positions=[(4, 0), (4, 8)])

    my_player = Player('Player 1', 4, 0, 2, RIGHT)
    my_player.automate = False

    cpu_player = Player('Player 2', 4, 8, 1, LEFT)
    cpu_player.automate = False

    env.register_players([my_player, cpu_player])

    speedup = 100
    user_mode = False
    input_cd = 0

    ##################

    tracker = {
        my_player : {'net' : dqn},
        cpu_player : {'net' : dqn_}
    }

    ##################
    def get_state(image_buffer_, view):
        image = torch.Tensor(env.screenshot(view, SIZE_W, SIZE_H))
        image = (image - 128) / (128 - 1)

        for k in range(len(image_buffer_) - 1):
            image_buffer_[k] = image_buffer_[k + 1]
        image_buffer_[len(image_buffer_) - 1] = image

        return torch.FloatTensor(image_buffer_[N_PLAYERS:])[None, :]


    ###################

    print('Sampling states using random policy...')
    random_states = torch.zeros(N_RANDOM_STATES, STACK_SIZE * N_PLAYERS, SIZE_W, SIZE_H)
    random_states = random_states.to(dqn.device)
    sampled = 0
    while sampled < N_RANDOM_STATES:
        env.reset()

        image_buffer = torch.zeros((STACK_SIZE + 1) * N_PLAYERS, SIZE_H, SIZE_W)
        for _ in range(FRAMES):
            env.update()

            env.render(my_player.view, display_score=False)
            state = get_state(image_buffer, my_player.view)
            if random.randint(0, 100) == 0:
                random_states[sampled] = state
                sampled += 1
                if sampled == N_RANDOM_STATES:
                    break

            env.perform_action(player=my_player, action=random.randint(0, NUM_ACTIONS - 1))

            env.update()
            env.render(my_player.view, display_score=False)
            state = get_state(image_buffer, my_player.view)
            if random.randint(0, 100) == 0:
                random_states[sampled] = state
                sampled += 1
                if sampled == N_RANDOM_STATES:
                    break

            env.perform_action(player=cpu_player, action=random.randint(0, NUM_ACTIONS - 1))

    avg_q_per_episode = defaultdict(list)
    reward_per_episode = defaultdict(list)
    score_per_episode = defaultdict(list)

    for i in range(EPISODES):
        ###
        #dqn_.eval_net.load_state_dict(dqn.eval_net.state_dict())
        ###

        env.reset(seed=i)

        p1 = env.players[env.ordering[0]]
        p2 = env.players[env.ordering[1]]

        image_buffer = torch.zeros((STACK_SIZE + 1) * N_PLAYERS, SIZE_H, SIZE_W)
        image_buffer_ = torch.zeros((STACK_SIZE + 1) * N_PLAYERS, SIZE_H, SIZE_W)
        action = action_ = None

        env.render(p1.view, display_score=False)

        for p in env.players:
            tracker[p]['reward'] = 0
            tracker[p]['episode_reward'] = 0
            tracker[p]['score'] = env.get_score(p)

        for frame_idx in range(FRAMES if not user_mode else 10 ** 10):
            input_cd = max(0, input_cd - 1)

            if speedup < 20 and not user_mode:
                time.sleep(1 / speedup)
            elif user_mode:
                time.sleep(1 / 60)

            if not user_mode:
                ###########################

                env.update()

                ###########################

                if frame_idx % FRAME_SKIP == 0:
                    env.render(p1.view, display_score=False)
                    dqn_input = get_state(image_buffer, p1.view)

                    env.render(p2.view, display_score=False)
                    get_state(image_buffer_, p2.view)

                    # calculate reward
                    if not env.kill:
                        reward = env.get_score(p1) - tracker[p1]['score']
                        #reward /= 100
                        if reward > 0:
                            reward = 1
                        elif reward < 0:
                            reward = -1
                    else:
                        reward = -1

                    tracker[p1]['score'] = env.get_score(p1)
                    tracker[p1]['episode_reward'] += reward

                    if tracker[p1]['net']:
                        s_last = torch.FloatTensor(image_buffer[:-N_PLAYERS])
                        s_current = torch.FloatTensor(image_buffer[N_PLAYERS:])
                        if action is not None:
                            tracker[p1]['net'].store_transition(s_last, action, reward, s_current)

                        action = tracker[p1]['net'].choose_action(dqn_input)

                        if tracker[p1]['net'].memory_counter >= BATCH_SIZE * 3:
                            tracker[p1]['net'].learn()
                    else:
                        action = random.randint(0, NUM_ACTIONS - 1)

                    env.perform_action(player=p1, action=action)

                    ###################################

                    if speedup < 10 and not user_mode:
                        time.sleep(1 / speedup)
                    elif user_mode:
                        time.sleep(1 / 60)

                env.update()

                if frame_idx % FRAME_SKIP == 0:
                    env.render(p2.view, display_score=False)
                    dqn_input = get_state(image_buffer_, p2.view)
                    env.render(p1.view, display_score=False)
                    get_state(image_buffer, p1.view)

                    # calculate reward
                    if not env.kill:
                        reward_ = env.get_score(p2) - tracker[p2]['score']
                        #reward_ /= 100
                        if reward_ > 0:
                            reward_ = 1
                        elif reward_ < 0:
                            reward_ = -1
                    else:
                        reward_ = -1

                    tracker[p2]['score'] = env.get_score(p2)
                    tracker[p2]['episode_reward'] += reward_

                    if tracker[p2]['net']:
                        s_last = torch.FloatTensor(image_buffer_[:-N_PLAYERS])
                        s_current = torch.FloatTensor(image_buffer_[N_PLAYERS:])
                        if action_ is not None:
                            tracker[p2]['net'].store_transition(s_last, action_, reward_, s_current)

                        action_ = tracker[p2]['net'].choose_action(dqn_input)

                        if tracker[p2]['net'].memory_counter >= BATCH_SIZE * 3:
                           tracker[p2]['net'].learn()
                    else:
                        action_ = random.randint(0, NUM_ACTIONS - 1)

                    env.perform_action(player=p2, action=action_)

            ###################################

            elif input_cd == 0:
                input_cd = 30
                for p in [my_player, cpu_player]:
                    env.update()
                    env.render(my_player.view, display_score=False)

                    while True:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False

                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_w]:
                            env.perform_action(p, MOVE_UP)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_s]:
                            env.perform_action(p, MOVE_DOWN)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_a]:
                            env.perform_action(p, MOVE_LEFT)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_d]:
                            env.perform_action(p, MOVE_RIGHT)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_SPACE]:
                            env.perform_action(p, EAT)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_n]:
                            env.perform_action(p, NOTHING)
                            time.sleep(.1)
                            break
                        elif keys[pygame.K_p]:
                            env.perform_action(p, PUNISH)
                            time.sleep(.1)
                            break

            keys = pygame.key.get_pressed()  # checking pressed keys
            if keys[pygame.K_e]:
                speedup = 100
            elif keys[pygame.K_q]:
                speedup = 10

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        print("episode: {}".format(i))
        for p in env.players:
            print('name {} reward {} score {}'.format(p.name, tracker[p]['episode_reward'], tracker[p]['score']))

            if tracker[p]['net']:
                eval_result = tracker[p]['net'].eval_net(random_states)
                avg_q_per_episode[p.name].append(eval_result.max(axis=1)[0].mean().item())

            reward_per_episode[p.name].append(tracker[p]['episode_reward'])
            score_per_episode[p.name].append(tracker[p]['score'])

            print(avg_q_per_episode[my_player.name][-1])

    for player in env.players:
        qs = avg_q_per_episode[player.name]
        print(qs)
        plt.plot(np.arange(len(qs)), qs, '-o', label=player.name)
        plt.xticks(np.arange(len(qs)))
        plt.xlabel('Episode')
        plt.ylabel('Average maximum Q-value')
        plt.title('Convergence of DQN agents')

    print(reward_per_episode[my_player.name])
    print(reward_per_episode[cpu_player.name])

    print(np.mean(reward_per_episode[my_player.name]))
    print(np.mean(reward_per_episode[cpu_player.name]))

    print(np.mean(score_per_episode[my_player.name]))
    print(np.mean(score_per_episode[cpu_player.name]))

    plt.legend()
    plt.show()

    #torch.save(dqn.eval_net.state_dict(), 'models/200_noclamp_twonet.pth') 4633 2108
    #torch.save(dqn.eval_net.state_dict(), 'models/200_clamp_twonet.pth') #4950 3475
    #torch.save(dqn.eval_net.state_dict(), 'models/200_clamp_onenet.pth') 4868 3583
    #torch.save(dqn.eval_net.state_dict(), 'models/300_clamp_onenet_double.pth') 5288  5305
    torch.save(dqn.eval_net.state_dict(), 'models/200_clamp_twonet_double.pth')