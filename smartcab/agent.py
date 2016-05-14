import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import math

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.Qtable = {}
        self.counter = 0  #counts number of steps learned
        self.steps_counter = 0
        self.gamma = 0  #discounting rate of future rewards
        self.epsilon = 0.9 / (1+(math.exp(-(self.counter-50))))
        """The Logistic function ranges from 0 to 0.9 as the number of total
        steps increases during the learning process.  At about 50 steps, the 'logistic'
        part of the curve will be implemented and random actions will give way to
        the 'best' action, gradually.  However, I have decided to limit the chance
        of choosing the 'best' action at 90%, as to introduce opportunity to break
        free from any local minimum Q_value(state, action) that may be present"""
        self.alpha = 0.5 #1 - ( 0.75 / (1 + math.exp(-(self.counter-200)))) #alpha ranges from 1 to 0.25
        """The learning rate will start at 1 and move towards 0.25 as the number of steps increases.
        A steep drop in learning rate will occur at about 200 steps."""
        self.reward_list = []
        self.action_list = []
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.steps_counter = 0
        print self.Qtable

    def update(self, t):
        #Remember the previous state
        if self.steps_counter != 0 :
            state_previous = self.state
            reward_previous = self.reward_list[-1]
            action_previous = self.action_list[-1]
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        Qtable = self.Qtable
        self.counter += 1

        # TODO: Update state
        #Sense the current state:
        self.state = (  ("directions",self.next_waypoint),
                        ("light",inputs['light']),
                        ("oncoming", inputs['oncoming']),
                        ("left",inputs['left']))

        # TODO: Select action according to your policy
        if Qtable.has_key(self.state) :  #check if state has been encountered before
            action_q = Qtable[self.state]
            if random.random() < self.epsilon :  #choose 'best' if epsilon is > than random float
                action = max(action_q, key = action_q.get)  #look for the argmax action; what if there are two actions with same reward?  Need a count command
            else :
                action = random.choice([None, 'forward', 'left', 'right'])
        else :
            Qtable.update({self.state : {None : 0, 'forward' : 0, 'left' : 0 , 'right' : 0}})
            action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward_list.append(reward)
        self.action_list.append(action)

        # TODO: Learn policy based on state, action, reward
        if self.steps_counter != 0 :
            """Bellman equation?"""
            Q_hat = (1-self.alpha)*Qtable[state_previous][action_previous] + \
            (self.alpha * (reward_previous + (self.gamma * max(Qtable[self.state].values()))))

            Qtable[state_previous][action_previous] = Q_hat

        self.steps_counter += 1
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=20)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
