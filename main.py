import sys
import gym
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor

# class of replay buffer, to store transitions we use deque data structure
class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = deque(maxlen=max_size)

    def new_tuple(self, state, action, reward, new_state, done):
        self.storage.append((state, action, reward, new_state, done))

    def get_minibatch(self, size):
        return random.sample(self.storage, size)

# function to initialize weights of nn
def nn_init_weight(size):
    weight = 1./np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-weight, weight)

# class of critic network, contains three hidden layers and output layer
# actions (input2) are included from the second hidden layer
class CriticNetwork(nn.Module):
    def __init__(self, input1, input2, hidden1, hidden2, hidden3, output, init_weight=3e-3):
        super().__init__()
        # state + action input
        self.input_layer = nn.Linear(input1, hidden1)
        self.input_layer.weight.data = nn_init_weight(self.input_layer.weight.data.size())

        self.hidden_layer1 = nn.Linear(hidden1+input2, hidden2)
        self.hidden_layer1.weight.data = nn_init_weight(self.hidden_layer1.weight.data.size())

        self.hidden_layer2 = nn.Linear(hidden2, hidden3)
        self.hidden_layer2.weight.data.uniform_(-init_weight, init_weight)

        self.out_layer = nn.Linear(hidden3, output)

    def forward(self, state, action):
        res1 = nn.functional.relu(self.input_layer(state))
        res2 = nn.functional.relu(self.hidden_layer1(torch.cat((res1, action), 1)))
        res3 = nn.functional.relu(self.hidden_layer2(res2))
        res4 = self.out_layer(res3)
        return res4

# actor network class, actor contains three layers
class ActorNetwork(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, scale, init_weight=3e-3):
        super().__init__()
        self.scale = scale

        self.input_layer = nn.Linear(input, hidden1)
        self.input_layer.weight.data = nn_init_weight(self.input_layer.weight.data.size())

        self.hidden_layer = nn.Linear(hidden1, hidden2)
        self.hidden_layer.weight.data = nn_init_weight(self.hidden_layer.weight.data.size())

        self.out_layer = nn.Linear(hidden2, output)
        self.out_layer.weight.data.uniform_(-init_weight, init_weight)

    def forward(self, state):
        res1 = nn.functional.relu(self.input_layer(state))
        res2 = nn.functional.relu(self.hidden_layer(res1))
        res3 = torch.tanh(self.out_layer(res2))
        res3 = res3 * self.scale
        return res3

# class to describe the behaviour of Uhlenbeck&Ornstein noise
class Noise:
    def __init__(self, mean, std_deviation, theta, dt):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.reset()

    def get_new_val(self):
        x = (self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt
             + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = np.zeros_like(self.mean)

# discretizing of action space [-1,1] with step 0.5
discrete_actions = [-1, -0.5, 0, 0.5, 1]

# function that represents the policy of DDPQ method
#  for continous version new action is the sum of ctor network output and noise
#  for discrete version policy includes Wolpertinger architecture,
#  but instead of k-nn method we use cycle to find 2 closest actions,
#  it is easier and more convenient way for double inverted pendulum action space
def policy(state, actor, critic, noise, flag):
    action = actor.forward(torch.FloatTensor(state))
    action = action.detach().numpy()[0]
    action = [action] + noise.get_new_val()

    # if discrete policy
    if flag:
        for i in range(1, len(discrete_actions)):
            if action < discrete_actions[i] and action > discrete_actions[i-1]:
                break
        st = torch.FloatTensor([state, state]).to(device)
        ac = torch.FloatTensor([[discrete_actions[i]], [discrete_actions[i-1]]]).to(device)
        res = critic.forward(st, ac)
        res = res.data.tolist()

        if res[0][0] > res[1][0]:
            return [discrete_actions[i]]
        else:
            return [discrete_actions[i-1]]
    return action


# DDPQ method
def ddpq(policy_type):
    # using flag to take into account the policy type
    if policy_type == "discrete":
        flag = True
    else:
        flag = False

    # noise init
    std_dev = 0.2 * np.ones(1)
    theta = 0.15
    dt = 0.1
    mean = np.zeros(1)
    noise = Noise(mean, std_dev, theta, dt)

    # init number of epoch and maximum of transitions for one simulation
    epoch_max = 4000
    trans_max = 500

    # neurons number for nn
    hidden1 = 256
    hidden2 = 256
    hidden3 = 256

    # size of minibatch from replay buffer
    minibatch_size = 128

    # ddpg parameters initialization
    gamma = 0.99
    tau = 0.001
    lr_actor = 0.0001
    lr_critic = 0.001

    # critic and actor networks init
    print('actor scale', actor_scale)
    actor = ActorNetwork(env.observation_space.shape[0], hidden1, hidden2, env.action_space.shape[0], actor_scale)
    a_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)

    critic = CriticNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden1, hidden2, hidden3,
                           env.action_space.shape[0])
    c_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    c_loss_func = nn.MSELoss()

    # target networks init
    target_actor = ActorNetwork(env.observation_space.shape[0], hidden1, hidden2, env.action_space.shape[0],
                                actor_scale)
    for a_par, t_a_par in zip(actor.parameters(), target_actor.parameters()):
        t_a_par.data.copy_(a_par.data)

    target_critic = CriticNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden1, hidden2, hidden3,
                                  env.action_space.shape[0])
    for a_par, t_a_par in zip(critic.parameters(), target_critic.parameters()):
        t_a_par.data.copy_(a_par.data)

    # init replay buffer with size 1000000 elements
    rb = ReplayBuffer(1000000)

    # init variables for plots
    rewards = []
    avg_rewards = []

    print('Start DDPQ')
    for i in range(0, epoch_max):
        state = env.reset()
        epoch_reward = 0
        noise.reset()
        for j in range(0, trans_max):
            # choosing of action and applying it to the environment
            action = policy(state, actor, critic, noise, flag)
            print('action', action)
            new_state, reward, done, _ = env.step(action)

            # calculating sum of rewards during the epoch
            epoch_reward += reward

            # add new transition to the replay buffer
            rb.new_tuple(state, action, reward, new_state, done)

            # if we are ready to update networks(enough transitions in R)
            if len(rb.storage) > minibatch_size:
                # obtaining elements of transitions
                sample = rb.get_minibatch(minibatch_size)
                st = []
                ac = []
                re = []
                st_new = []
                dn = []
                for trans in sample:
                    s, u, r, snew, d = trans
                    st.append(s)
                    ac.append(u)
                    re.append([r])
                    st_new.append(snew)
                    dn.append(d)
                # creation of tensors
                st = torch.FloatTensor(st).to(device)
                ac = torch.FloatTensor(ac).to(device)
                re = torch.FloatTensor(re).unsqueeze(1).to(device)
                dn = torch.FloatTensor(np.float32(dn)).unsqueeze(1).to(device)
                st_new = torch.FloatTensor(st_new).to(device)

                # calculate Yi, we add (1.0 - dn) coefficient, to do not count actions that lead to terminal state
                u_target = target_actor.forward(st_new)
                q_target = target_critic.forward(st_new, u_target)
                Yi = re + (1.0 - dn) * gamma * q_target.detach()

                # updating of weights is weird from the first glance, but it is the pytorch way to do it
                # https://towardsdatascience.com/up-and-running-with-pytorch-minibatching-dataloading-and-model-building-7c3fdacaca40

                # update critic
                Q = critic.forward(st, ac)
                c_optimizer.zero_grad()
                critic_loss = c_loss_func(Q, Yi)
                critic_loss.backward()
                c_optimizer.step()

                # update actor
                # we use negative sign here to maximize the value
                a_optimizer.zero_grad()
                J = -critic.forward(st, actor.forward(st))
                J = J.mean()
                J.backward()
                a_optimizer.step()

                # update target
                for (a, b) in zip(target_critic.parameters(), critic.parameters()):
                    a.data.copy_(b * tau + a * (1 - tau))
                for (a, b) in zip(target_actor.parameters(), actor.parameters()):
                    a.data.copy_(b * tau + a * (1 - tau))

            if done:
                print('simulation done in', j, 'steps')
                break
            state = new_state
        rewards.append(epoch_reward)
        # rewards.append(epoch_reward/(j + 1)) # with discout reward
        avg_rewards.append(np.mean(rewards[-10:]))
        print('epoch ', i, ' is finished, reward', epoch_reward)

    # plot creation
    plt.plot(rewards)
    plt.plot(avg_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()


# FQI for ExtraTreesRegressor
def fqi():
    # choosing parameters to create a dataset
    epoch_max = 1000
    trans_max = 500

    # noise initialization
    std_dev = 0.2 * np.ones(1)
    theta = 0.15
    dt = 0.1
    mean = np.zeros(1)
    noise = Noise(mean, std_dev, theta, dt)

    data = list()
    rewards = []
    # creating of data set
    for i in range(0, epoch_max):
        state = env.reset()
        epoch_reward = 0
        noise.reset()
        for j in range(0, trans_max):
            # random policy action
            action = env.action_space.sample()
            print('action', action)
            new_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            new_line = list()
            new_line.append(state)
            new_line.append(action[0])
            new_line.append(reward)
            new_line.append(new_state)
            data.append(new_line)
            if done:
                break
            state = new_state
        rewards.append(epoch_reward)
        # rewards.append(epoch_reward/(j + 1)) # with discout reward

    # steps represent the depth of FQI
    steps = 100
    discount_factor = 0.95
    approximation = list()
    for i in range(1, steps + 1):
        print(str(i) + ' out of ' + str(steps))
        x, y = [], []
        for line in data:
            ps = line[0].tolist()
            action = line[1]
            reward = line[2]
            new_ps = line[3].tolist()
            # X contains all states vars + action
            x.append([ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10], action])
            if i == 1:
                y.append(reward)
            else:
                y.append(reward + discount_factor *
                         max(
                             approximation[-1].predict(np.array([new_ps[0], new_ps[1], new_ps[2], new_ps[3], new_ps[4], new_ps[5], new_ps[6], new_ps[7], new_ps[8], new_ps[9], new_ps[10], -1]).reshape(1, -1))[0],
                             approximation[-1].predict(np.array([new_ps[0], new_ps[1], new_ps[2], new_ps[3], new_ps[4], new_ps[5], new_ps[6], new_ps[7], new_ps[8], new_ps[9], new_ps[10], -0.5]).reshape(1, -1))[0],
                             approximation[-1].predict(np.array([new_ps[0], new_ps[1], new_ps[2], new_ps[3], new_ps[4], new_ps[5], new_ps[6], new_ps[7], new_ps[8], new_ps[9], new_ps[10], 0]).reshape(1, -1))[0],
                             approximation[-1].predict(np.array([new_ps[0], new_ps[1], new_ps[2], new_ps[3], new_ps[4], new_ps[5], new_ps[6], new_ps[7], new_ps[8], new_ps[9], new_ps[10], 0.5]).reshape(1, -1))[0],
                             approximation[-1].predict(np.array([new_ps[0], new_ps[1], new_ps[2], new_ps[3], new_ps[4], new_ps[5], new_ps[6], new_ps[7], new_ps[8], new_ps[9], new_ps[10], 1]).reshape(1, -1))[0]))
        approximation.append(ExtraTreesRegressor(n_estimators=10).fit(np.array(x), np.array(y)))

    # simulation to test FQI results
    for i in range(0, epoch_max):
        state = env.reset()
        epoch_reward = 0
        noise.reset()
        for j in range(0, trans_max):
            bestQ = approximation[-1].predict(np.array([state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8],state[9],state[10], -2]).reshape(1, -1))[0]
            bestact = -1
            for action in discrete_actions:
                Q = approximation[-1].predict(np.array([state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8],state[9],state[10], action]).reshape(1, -1))[0]
                if bestQ < Q:
                    bestQ = Q
                    bestact = action
            # print('action', bestact)
            new_state, reward, done, _ = env.step(bestact)
            if done:
                break
            epoch_reward += reward
            state = new_state
        print('epoch', i, 'finished')
        rewards.append(epoch_reward)
        # rewards.append(epoch_reward / (j + 1)) # with discount reward
    # creating of plot both for random policy and wth policy based on FQI results
    plt.plot(rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.show()


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print('Use device ', device)

    env = gym.make('InvertedDoublePendulum-v2')
    print('Action space', env.action_space.low, env.action_space.high)
    print('Action numbers', env.action_space.shape[0])
    print('Observational vars: ', env.observation_space.shape[0])
    actor_scale = env.action_space.high[0]
    print(actor_scale, 'actor scale')
    ddpq("discrete")
    ddpq("continuous")
    fqi()
