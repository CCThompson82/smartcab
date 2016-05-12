import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.Qtable = {}

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        #dont delete qtable values if re-running simulations ??

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        Qtable = self.Qtable
        # TODO: Update state

        self.state = (("directions",self.next_waypoint),("light",inputs['light']), ("oncoming", inputs['oncoming']), ("left",inputs['left']))

        # TODO: Select action according to your policy
        if Qtable.has_key(self.state) :  #check if state has been encountered before
            action_q = Qtable[self.state]
            action = max(action_q, key = action_q.get)  #look for the argmax action
        else :
            Qtable.update({self.state : {None : 0, 'forward' : 0, 'left' : 0 , 'right' : 0}})
            action = random.choice([None, 'forward', 'left', 'right'])

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        Q_state_action_prime = 0 # """How does one know what the next state will be in order to use Q-learning convergence??)"""
        Q_hat = (1-0.5)*Qtable[self.state][action] + (0.5 * (reward + (0 * Q_state_action_prime)))  #alpha = 0.5 to start; gamma = 0
        Qtable[self.state][action] = Q_hat
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=2)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
