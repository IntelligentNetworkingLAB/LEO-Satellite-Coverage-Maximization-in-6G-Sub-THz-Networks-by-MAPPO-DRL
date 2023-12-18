import random as rand
from scipy.spatial import distance
import math
import numpy as np
from matplotlib import pyplot as plt

diff_x = [1, 1, 0, -1, -1, -1 ,0, 1]
diff_y = [0, 1, 1, 1, 0, -1, -1, -1]

class EnvironmentRoute():
    def __init__(self):
        self.max_range = 3000
        self.X_num = 30
        self.Y_num = 10
        self.max_step = 500
        self.reset()        
        return
    
    def getnewpoint(self):
        rand_pos = [rand.randint(0, self.X_num - 1), rand.randint(0, self.Y_num - 1)]
        while self.ConstellGraph[rand_pos[0]][rand_pos[1]][2] != 0:
            rand_pos = [rand.randint(0, self.X_num - 1), rand.randint(0, self.Y_num - 1)]
        self.ConstellGraph[rand_pos[0]][rand_pos[1]][2] = 1
        return rand_pos     

    def whereendpos(self):
        tmp_distance = -1 
        yy = int(self.desire_pos[1]/100)
        for x in range(self.X_num):
            for y in range(yy, yy + 1):
                comp = distance.euclidean([self.ConstellGraph[x][y][0], self.ConstellGraph[x][y][1]], [self.desire_pos[0], self.desire_pos[1]])
                if tmp_distance == -1 or tmp_distance > comp:
                    tmp_distance = comp
                    self.end_pos = [x, y]
        return
    
    def set_obstacles(self, _obs):
        self.obs_poss = _obs
        for ob in _obs:
            self.ConstellGraph[ob[0]][ob[1]][2] = 1

    def set_observation(self):
        observation = []
        observation.extend(self.cur_pos)
        for obs in self.obs_poss:
            observation.extend(obs)
        observation.extend(self.end_pos)
        return observation

    def reset(self):
        self.isTerm = False
        self.cur_step = 0
        self.cur_reward = 0
        self.ConstellGraph = []
        for x in range(self.X_num):
            tmpRow = []
            for y in range(self.Y_num):
                tmpRow.append([x*300, y*100, 0])
            self.ConstellGraph.append(tmpRow)

        self.cur_pos = [rand.randint(0, self.X_num - 1), rand.randint(0, self.Y_num - 1)]
        self.desire_pos = [3000*rand.random(), 1000*rand.random()]
        self.whereendpos()
        while self.cur_pos[0] == self.end_pos[0] and self.cur_pos[1] == self.end_pos[1]:
            self.desire_pos = [3000*rand.random(), 1000*rand.random()]
            self.whereendpos()

        self.ConstellGraph[self.cur_pos[0]][self.cur_pos[1]][2] = 1
        self.ConstellGraph[self.end_pos[0]][self.end_pos[1]][2] = 1

        self.obs_poss = []
        self.obs_poss.append(self.getnewpoint())
        self.obs_poss.append(self.getnewpoint())

        return self.set_observation()
    
    def get_step_route(self):
        # state, reward, isTerm
        if self.cur_step == self.max_step or self.isTerm == True:
            self.cur_reward = -1
        elif self.cur_pos == self.end_pos:
            self.cur_reward = 1
        else:
            self.cur_reward = 1.0 / distance.euclidean([self.ConstellGraph[self.cur_pos[0]][self.cur_pos[1]][0], self.ConstellGraph[self.cur_pos[0]][self.cur_pos[1]][1]],
                                                       [self.ConstellGraph[self.end_pos[0]][self.end_pos[1]][0], self.ConstellGraph[self.end_pos[0]][self.end_pos[1]][1]])
        return [self.set_observation(), self.cur_reward, self.isTerm, {}]

    def checkin(self, _x, _y):
        if _x >= 0 and _x < self.X_num and _y >= 0 and _y < self.Y_num and self.ConstellGraph[_x][_y][2] == 0:
            return True
        return False

    def set_action_route(self, action):
        if self.checkin(self.cur_pos[0] + diff_x[action], self.cur_pos[1] + diff_y[action]):
            self.ConstellGraph[self.cur_pos[0]][self.cur_pos[1]][2] = 0
            self.cur_pos = [self.cur_pos[0] + diff_x[action], self.cur_pos[1] + diff_y[action]]
            self.ConstellGraph[self.cur_pos[0]][self.cur_pos[1]][2] = 1
        else:
            self.isTerm = True
        return

    def step_route(self):
        self.cur_step += 1
        for x in range(self.X_num):
            for y in range(self.Y_num):
                self.ConstellGraph[x][y][0] += 10
                if(self.ConstellGraph[x][y][0] > self.max_range):
                    tmpRow = []
                    for yy in range(self.Y_num):
                        tmpRow.append([0, yy*100, self.ConstellGraph[x][yy][2]])
                    self.ConstellGraph.insert(0, tmpRow)
                    self.ConstellGraph.pop(x+1)
                    break
        
        self.ConstellGraph[self.end_pos[0]][self.end_pos[1]][2] = 0
        self.whereendpos()
        self.ConstellGraph[self.end_pos[0]][self.end_pos[1]][2] = 2

        for obs in self.obs_poss:
            self.ConstellGraph[obs[0]][obs[1]][2] = 0

        self.obs_poss = []
        self.obs_poss.append(self.getnewpoint())
        self.obs_poss.append(self.getnewpoint())
        return

class EnvironmentPhase():
    def __init__(self):
        self.rf = 0.8
        self.max_step = 100
        self.reset()        
        return
    
    def reset(self):
        self.cur_reward = 0
        self.isTerm = False
        self.step_count = 0
        self.phase = [ 2*math.pi*np.random.rand() for _ in range(10)]
        self.LoS = complex(np.random.randn(), np.random.randn())
        self.NLoS = complex(np.random.randn(), np.random.randn())
        self.channel = math.sqrt(self.rf/(self.rf + 1))*self.LoS + math.sqrt(1/(self.rf + 1))*self.NLoS
        return self.set_observation()
    
    def set_observation(self):
        observation = []
        observation.append(self.LoS.imag)
        observation.append(self.LoS.real)
        observation.append(self.NLoS.imag)
        observation.append(self.NLoS.real)
        return observation
    
    def set_action(self, _act):
        self.phase = 2 * math.pi * _act
        m = [ np.exp(1j*self.phase[i]) for i in range(len(self.phase))]
        m = np.array(m)
        S = np.diag(m) # M x M
        self.cur_reward = np.sum((S.dot(self.channel))**2).real
        return
    
    def step(self):
        self.step_count += 1
        return
        
    def get_step(self):
        if self.step_count == self.max_step:
            self.isTerm = True
        return [self.set_observation(), self.cur_reward, self.isTerm, {}]

if __name__ == '__main__':
    env = EnvironmentPhase()
    pp = 0
    resultX = []
    for _ in range(5000):
        m = [ np.exp(1j*pp) for _ in range(10)]
        m = np.array(m)
        S = np.diag(m) # M x M
        print(np.sum(((m.dot(env.channel)))**2)**2)
        resultX.append(np.sum((m.dot(env.channel))**2)**2)
        pp += 0.001
    
    plt.plot(resultX)
    plt.show()