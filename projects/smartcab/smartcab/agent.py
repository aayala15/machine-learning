import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

import numpy as np
import matplotlib.pyplot as plt

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Qvalues = {}
        self.state = None
        self.action = None
        self.reward = 0.0
        self.alpha = 0.9
        self.gamma = 0.5
        self.epsilon = 0.05
        self.n = -1
        self.time = np.zeros(100)
        self.goal = np.zeros(100)
        self.right = np.zeros(100)
        self.wrong = np.zeros(100)
        
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.state != None:
            x = self.state + (('action', self.action),)
            Qvalue = 0
            if x in self.Qvalues.keys():
                Qvalue = self.Qvalues[x]
            self.Qvalues[x] = (1-self.alpha)*Qvalue + self.alpha*self.reward 
        self.state = None
        self.action = None
        self.reward = 0.0
        self.n = self.n + 1
        
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # TODO: Update state
        if inputs['light'] == 'red':
            inputs['oncoming'] = None
            inputs['right'] = None
        else:
            inputs['left'] = None
            inputs['right'] = None
        
        state_0 = self.state
        action_0 = self.action
        
        state_1 = tuple(inputs.items()) + (('next_waypoint', self.next_waypoint),)
        self.state = state_1
        
        # TODO: Select action according to your policy
        action_set = [None, 'forward', 'left', 'right']
        
        action_1 = self.next_waypoint
        x_1 = state_1 + (('action', action_1),)
        Qvalue_1 = 0
        if x_1 in self.Qvalues.keys():
            Qvalue_1 = self.Qvalues[x_1]
        
        for a in [x for x in action_set if x != self.next_waypoint]:
            x_1 = state_1 + (('action', a),)
            if x_1 in self.Qvalues.keys():
                if self.Qvalues[x_1] > Qvalue_1:
                    Qvalue_1 = self.Qvalues[x_1]
                    action_1 = a
        
        if random.random() < self.epsilon:
            action_1 = random.choice([None, 'forward', 'left', 'right'])
        action = action_1
        self.action = action
        # print("action = {}, next_waypoint = {}".format(action, self.next_waypoint))
        
        # Execute action and get reward
        reward_0 = self.reward
        reward = self.env.act(self, action)
        self.reward = reward
        
        # TODO: Learn policy based on state, action, reward
        if state_0 != None:
            x_0 = state_0 + (('action', action_0),)
            Qvalue_0 = 0
            if x_0 in self.Qvalues.keys():
                Qvalue_0 = self.Qvalues[x_0]
            self.Qvalues[x_0] = (1-self.alpha)*Qvalue_0 + self.alpha*(reward_0 + self.gamma*Qvalue_1)
        
        self.time[self.n] = self.time[self.n] + 1
        if reward < 0:
            self.wrong[self.n] = self.wrong[self.n] + 1
        elif reward > 0:
            self.right[self.n] = self.right[self.n] + 1
        if reward >= 10 and deadline > -1:
            self.goal[self.n] = 1
        print("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward))  # [debug]


def run():
    """Run the agent for a finite number of trials."""
    
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    
    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
    
    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    fig, ax = plt.subplots()
    ax.plot(range(100), a.goal, 'o')
    ax.set_title('Does the agent reach the destination in time?')
    plt.savefig('destination.png')
    
    fig, ax = plt.subplots()
    ax.plot(range(100), a.right / a.time, 'o', color='g')
    ax.plot(range(100), a.wrong / a.time, 'o', color='r')
    plt.legend(('Right', 'Wrong'))
    ax.set_title('Traffic-rules')
    plt.savefig('traffic_rules.png')
    

if __name__ == '__main__':
    run()
