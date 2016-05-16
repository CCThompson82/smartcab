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
        self.Qtable = {}  #empty Qtable to be filled during update
        self.lesson_counter = 0  #counts number of steps learned
        self.steps_counter = 0 #counts steps in the trial

        #self.Q_init = 0  #initial Q^ values for new state-actions not observed yet.
        self.Q_init = 13 #initial Q^ values for new state-actions not observed yet.

        self.gamma = 0
        #self.gamma = 0.1  #discounting rate of future rewards

        self.epsilon = 1
        #self.epsilon = 0.75 + (0.24 / (1+( math.exp(-0.1*(self.lesson_counter-40)))))
        """The output for the Logistic function for epsilon ranges from 0.75 to 0.99, and increases as the number of total
        steps increases during the learning process.  Random actions will give way to
        the 'best' action, gradually, but will never exceed 99%."""
        self.alpha = 1
        #self.alpha = 1 - ( 0.5 / (1 + math.exp(-0.05*(self.lesson_counter-100)))) #alpha ranges from 1 to 0.5

        self.reward_previous = None
        self.action_previous = None
        self.state_previous = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.steps_counter = 0
        #print self.Qtable [not needed any longer]

    def update(self, t):
        # Gather inputs for current state
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.lesson_counter += 1  #count steps in total run

        #Define the current state based on inputs sensed
        self.state = (  ("directions",self.next_waypoint),
                        ("light",inputs['light']),
                        ("oncoming", inputs['oncoming']),
                        ("left",inputs['left']))

        # TODO: Select action according to your policy
        Qtable = self.Qtable #cover up Qtable variable with current object.Qtable feature to make syntax easier in this suite
        if Qtable.has_key(self.state) :  #check if state has been encountered before or not
            if random.random() < self.epsilon :  #if epsilon is not eclipsed by a random float, then choose the action with the largest Q^.  If epsilon is 1, then best option is always chosen as it cannot be eclipsed
                #pull the best action, or best actions if there are more than one with a max Q^ value
                argmax_actions = {action:Qhat for action, Qhat in Qtable[self.state].items() if Qhat == max(Qtable[self.state].values())}
                action = random.choice(argmax_actions.keys())  #note if only 1 action in this list, then it is only choice for random.choice
            else : # if random float eclipses epsilon, choose a random action.
                action = random.choice([None, 'forward', 'left', 'right'])
        else :  #state has never been encountered
            Qtable.update({self.state : {None : self.Q_init, 'forward' : self.Q_init, 'left' : self.Q_init, 'right' : self.Q_init}}) #Add state to Qtable dictionary
            action = random.choice([None, 'forward', 'left', 'right'])  #choose one of the actions at random

        # Execute action and get reward
        reward = self.env.act(self, action)  #what was the reward for the chosen action?

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # TODO: Learn policy based on state, action, reward
        if self.steps_counter > 0 :  #make sure it is not the first step in a trial.
            Q_hat = Qtable[self.state_previous][self.action_previous]
            Q_hat = Q_hat + (self.alpha * (self.reward_previous + (self.gamma * (max(Qtable[self.state].values()))) - Q_hat))
            Qtable[self.state_previous][self.action_previous] = Q_hat
            self.Qtable = Qtable
        #Store actions, state and reward as previous_ for use in the next cycle
        self.state_previous = self.state
        self.action_previous = action
        self.reward_previous = reward
        self.steps_counter += 1

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.1)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

if __name__ == '__main__':
    run()
