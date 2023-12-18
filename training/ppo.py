# 라이브러리 불러오기
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
import environment

# 파라미터 값 세팅 
state_size = 26
action_size = 20
hidden_size = 128

load_model = False
train_mode = True

discount_factor = 0.0
actor_lr = 1e-4
critic_lr = 5e-4
n_step = 256
batch_size = 128
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

run_step = 400000 if train_mode else 0
test_step = 1000

print_interval = 10
save_interval = 100

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/PPO/{date_time}"
load_path = f"./saved_models/PPO/20220502131128"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ActorCritic 클래스 -> Actor Network, Critic Network 정의 
class ActorNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ActorNetwork, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, hidden_size)
        self.d2 = torch.nn.Linear(hidden_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, action_size)
        self.std = torch.nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        mu = torch.clamp(self.mu(x), min=-5.0, max=5.0)
        std = torch.tanh(self.std(x)).exp()
        return mu, std
    
class CriticNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super(CriticNetwork, self).__init__(**kwargs)
        self.d1 = torch.nn.Linear(state_size, hidden_size)
        self.l = torch.nn.Linear(hidden_size + action_size, hidden_size)
        self.v = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x, x0):
        x = F.relu(self.d1(x))
        x = torch.cat((x, x0), dim=-1)
        x = F.relu(self.l(x))
        return self.v(x)

# PPOAgent 클래스 -> PPO 알고리즘을 위한 다양한 함수 정의 
class PPOAgent:
    def __init__(self):
        self.actor_network = ActorNetwork().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_network = CriticNetwork().to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.memory = list()
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor_network.load_state_dict(checkpoint["actor_network"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_network.load_state_dict(checkpoint["critic_network"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # 정책을 통해 행동 결정 
    def get_action(self, state, training=True):
        # 네트워크 모드 설정
        self.actor_network.train(training)

        mu, std = self.actor_network(torch.FloatTensor(state).to(device))
        z = torch.normal(mu, std) if training else mu
        return torch.tanh(z).cpu().detach().numpy()

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        self.actor_network.train()
        self.critic_network.train()

        state      = np.stack([m[0] for m in self.memory], axis=0)
        action     = np.stack([m[1] for m in self.memory], axis=0)
        reward     = np.stack([m[2] for m in self.memory], axis=0)
        next_state = np.stack([m[3] for m in self.memory], axis=0)
        done       = np.stack([m[4] for m in self.memory], axis=0)
        self.memory.clear()

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])
        # prob_old, adv, ret 계산 
        with torch.no_grad():
            mu, std  = self.actor_network(state)
            value = self.critic_network(state, action)
            m = Normal(mu, std)
            z = torch.atanh(torch.clamp(action, -1 + 1e-7, 1 - 1e-7))
            prob_old = m.log_prob(z)
            
            next_mu, next_std = self.actor_network(next_state)
            next_value = self.critic_network(next_state, torch.tanh(torch.normal(next_mu, next_std)))
            delta = reward + (1 - done) * discount_factor * next_value - value
            adv = delta.clone()
            adv, done = map(lambda x: x.view(n_step, -1).transpose(0,1).contiguous(), [adv, done])
            for t in reversed(range(n_step-1)):
                adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t+1]
            adv = adv.transpose(0,1).contiguous()
            
            ret = adv.view(-1, 1) + value
            adv = (adv - adv.mean()) / (adv.std() + 1e-7)
            adv = adv.view(-1, 1)

        # 학습 이터레이션 시작
        actor_losses, critic_losses = [], []
        idxs = np.arange(len(reward))
        for _ in range(n_epoch):
            np.random.shuffle(idxs)
            for offset in range(0, len(reward), batch_size):
                idx = idxs[offset : offset + batch_size]

                _state, _action, _value, _ret, _adv, _prob_old =\
                    map(lambda x: x[idx], [state, action, value, ret, adv, prob_old])
                
                mu, std = self.actor_network(_state)
                value_pred = self.critic_network(_state, _action)
                m = Normal(mu, std)
                z = torch.atanh(torch.clamp(_action, -1 + 1e-7, 1 - 1e-7))
                prob = m.log_prob(z)

                entropy_loss = -m.entropy().mean()

                # 가치신경망 손실함수 계산
                '''
                critic_loss = F.mse_loss(value_pred, _ret)
                '''
                value_pred_clipped = _value + torch.clamp(
                    value_pred - _value, -epsilon, epsilon
                )
                critic_loss1 = F.mse_loss(value_pred, _ret)
                critic_loss2 = F.mse_loss(value_pred_clipped, _ret)
                critic_loss = torch.max(critic_loss1, critic_loss2).mean() + 0.01 * entropy_loss


                self.critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()

                # 정책신경망 손실함수 계산
                ratio = (prob - _prob_old).sum(1, keepdim=True).exp()
                #ratio = prob / (_prob_old + 1e-7)
                surr1 = ratio * _adv
                surr2 = torch.clamp(ratio, min=1-epsilon, max=1+epsilon) * _adv
                actor_loss = -torch.min(surr1, surr2).mean() + 0.01 * entropy_loss
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor_network" : self.actor_network.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic_network" : self.critic_network.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록 
    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

# Main 함수 -> 전체적으로 PPO 알고리즘을 진행 
if __name__ == '__main__':
    env = environment.Environment()
    env.reset()
    step_info = env.get_steps()

    # PPO 클래스를 agent로 정의 
    agent = PPOAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
        
        state = step_info[0]
        action = agent.get_action(state, train_mode)
        env.set_action_ppo(action)
        env.step()

        # 환경으로부터 얻는 정보
        step_info = env.get_steps()
        next_state = step_info[0]
        reward = step_info[1]
        done = step_info[2]
        score += reward

        if train_mode:
            agent.append_sample(state, action, [reward], next_state, [done])
            # 학습수행
            if (step+1) % n_step == 0:
                actor_loss, critic_loss = agent.train_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        if done:
            env.reset()
            episode +=1
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록 
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses)  if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +\
                      f"Actor loss: {mean_actor_loss:.4f} / Critic loss: {mean_critic_loss:.4f}" )

            # 네트워크 모델 저장 
            if train_mode and episode % save_interval == 0:
                agent.save_model()