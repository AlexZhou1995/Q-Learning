import numpy as np
import random
from tqdm import tqdm

class GridWord(object):
	def __init__(self,x,y,goalR):
		self.board = -1*np.ones([x,y])
		self.board[x-1][y-1]=goalR
		self.pos = np.array([0,0])
		self.x = x
		self.y = y
		self.action_set = {1,2,3,4}
	
	def getStateDim(self):
		return self.x*self.y

	def getActionDim(self):
		return len(self.action_set)

	def new_game(self):
		self.pos = np.array([0,0])
		return self.pos.copy(),0,{2,4},False
	
	def step(self, action, is_train=True):
		#1,2,3,4 is up,down,left,right. No loop
		if action==1:
			self.pos[0]-=1
		elif action==2:
			self.pos[0]+=1
		elif action==3:
			self.pos[1]-=1
		elif action==4:
			self.pos[1]+=1
		else:
			print ('action error')

		a_set={1,2,3,4}
		if self.pos[0]==0:
			a_set.remove(1)
		if self.pos[0]==self.x-1:
			a_set.remove(2)
		if self.pos[1]==0:
			a_set.remove(3)
		if self.pos[1]==self.y-1:
			a_set.remove(4)

		end = False
		if list(self.pos)==[self.x-1,self.y-1]:
			end=True

		return self.pos.copy(), self.board[self.pos[0]][self.pos[1]], a_set, end

class Agent(object):
	def __init__(self, env, gamma=0.9, ep_start=1.0, ep_end=0.01, t_ep_end=100,t_learn_start=5):
		self.env = env
		self.Q = np.zeros([self.env.getStateDim(),self.env.getActionDim()])
		self.ep_start = ep_start
		self.ep_end = ep_end
		self.t_ep_end = t_ep_end
		self.t_learn_start = t_learn_start
		self.gamma = gamma
		#self.state = np.array([0,0])
	def train(self,iter):
		state,reward,action_set,end = env.new_game()
		#self.state = state
		stepNum=0
		for i in tqdm(range(iter)):
			#ep = (self.ep_end + max(0., (self.ep_start - self.ep_end) * (self.t_ep_end - max(0., i - self.t_learn_start)) / self.t_ep_end))
			ep = 0.99
			#1.chooseAction
			action = self.chooseAct(state, ep, action_set)
			#2.act
			state_next,reward,action_set,end = self.env.step(action)
			#3.observe
			q = self.observe(state,state_next,reward,action,end)

			state = state_next
			stepNum+=1

			if end:
				state,reward,action_set,end = env.new_game()
				stepNum=0

	def play(self):
		state,reward,action_set,end = env.new_game()
		stepNum = 0
		while(True):
			ep = 0
			#1.chooseAction
			action = self.chooseAct(state, ep, action_set)
			#2.act
			state_next,reward,action_set,end = self.env.step(action)

			print('from (%d,%d) to (%d,%d)'%(state[0],state[1],state_next[0],state_next[1]))

			state = state_next
			stepNum+=1

			if end:
				break

	def predict(self,state,action_set):
		idx = self.env.y*state[0] + state[1]
		act=0
		val = -999999
		for each in action_set:
			if val < self.Q[idx,each-1]:
				val = self.Q[idx,each-1]
				act = each
		return act


	def chooseAct(self,state,ep,action_set):
		if random.random() < ep:
			return random.sample(action_set,1)[0]
		else:
			return self.predict(state,action_set)

	def observe(self,state,state_next,reward,action,end):
		idx = self.env.y*state[0] + state[1]
		idx_next = self.env.y*state_next[0] + state_next[1]
		self.Q[idx,action-1] = reward + self.gamma*max(self.Q[idx_next,:])
		return np.mean(self.Q)

	def printQ(self):
		print(self.Q)

if __name__ == '__main__':
	env = GridWord(5,5,100)
	angent = Agent(env)
	angent.train(10000)
	angent.printQ()
	angent.play()

