import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import environment

# DDPG를 위한 파라미터 값 세팅
state_size = 26
action_size = 50
hidden_size = 128

load_model = True
train_mode = False

batch_size = 128
mem_maxlen = 30000
discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

run_step = 400000 if train_mode else 0
test_step = 1000
train_start_step = 5000

print_interval = 10
save_interval = 100

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/DDPG/{date_time}"
load_path = f"./saved_models/DDPG/20230413230454"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 클래스 -> DDPG Actor 클래스 정의
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return  torch.tanh(self.mu(x))

# Critic 클래스 -> DDPG Critic 클래스 정의
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size+action_size, hidden_size)
        self.q = torch.nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat((x, action), dim=-1)
        x = torch.relu(self.fc2(x))
        return self.q(x)

# DDPGAgent 클래스 -> DDPG 알고리즘을 위한 다양한 함수 정의
class DDPGAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # OU noise 기법에 따라 행동 결정
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.actor.train(training)
        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return [action]

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        # Critic 업데이트
        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = reward + (1 - done) * discount_factor * next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # 소프트 타겟 업데이트를 위한 함수
    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록
    def write_summray(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)


# Main 함수 -> 전체적으로 DDPG 알고리즘을 진행
if __name__ == '__main__':
    env = environment.Environment()
    env.reset()
    step_info = env.get_steps()

    # DDPGAgent 클래스를 agent로 정의
    agent = DDPGAgent()

    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False

        state = step_info[0]
        action = agent.get_action(state, train_mode)
        env.set_action(action[0])
        env.step()

        # 환경으로부터 얻는 정보
        step_info = env.get_steps()
        next_state = step_info[0]
        reward = step_info[1]
        done = step_info[2]
        score += reward
        if train_mode:
            agent.append_sample(state, action[0], [reward], next_state, [done])

        if train_mode and step > max(batch_size, train_start_step):
            # 학습 수행
            actor_loss, critic_loss = agent.train_model()
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            # 타겟 네트워크 소프트 업데이트
            agent.soft_update_target()

        if done:
            env.reset()
            episode += 1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses)
                mean_critic_loss = np.mean(critic_losses)
                agent.write_summray(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Actor loss: {mean_actor_loss:.4f} / Critic loss: {mean_critic_loss:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()
