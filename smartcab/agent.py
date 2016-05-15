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
        self.lesson_counter = 0  #counts number of steps learned
        self.steps_counter = 0
        self.gamma = 0  #discounting rate of future rewards
        #self.gamma = 1
        self.epsilon = 1
        #self.epsilon = 0.9 / (1+( e^(-(self.lesson_counter-50))))
        """The Logistic function ranges from 0 to 0.9 as the number of total
        steps increases during the learning process.  At about 50 steps, the 'logistic'
        part of the curve will be implemented and random actions will give way to
        the 'best' action, gradually.  However, I have decided to limit the chance
        of choosing the 'best' action at 90%, as to introduce opportunity to break
        free from any local minimum Q_value(state, action) that may be present"""
        self.alpha = 1
        #self.alpha = 1 - ( 0.75 / (1 + e^(-(self.counter-200)))) #alpha ranges from 1 to 0.25
        """The learning rate will start at 1 and move towards 0.25 as the number of steps increases.
        A steep drop in learning rate will occur at about 200 steps."""
        self.reward_previous = None
        self.action_previous = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.steps_counter = 0
        print self.Qtable

    def update(self, t):
        # Gather inputs for current state
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.lesson_counter += 1

        #Define the current state based on inputs sensed
        self.state = (  ("directions",self.next_waypoint),
                        ("light",inputs['light']),
                        ("oncoming", inputs['oncoming']),
                        ("left",inputs['left']))

        # TODO: Select action according to your policy
        Qtable = self.Qtable #cover up Qtable variable with current object.Qtable feature to make syntax easier in this suite
        if Qtable.has_key(self.state) :  #check if state has been encountered before or not
            if random.random() < self.epsilon :  #if epsilon is not eclipsed by a random float, then choose the action with the largest Qhat.  If epsilon is 1, then best option is always chosen as it cannot be eclipsed
                #pull the best action, or best actions if there are more than one with a max Qhat value
                argmax_actions = {action:Qhat for action, Qhat in Qtable[self.state].items() if Qhat == max(Qtable[self.state].values())}
                action = random.choice(argmax_actions.keys())
            else : # if random float eclipses epsilon, choose a random action.  New feature idea: choose an action that is not the current argmax action
                action = random.choice([None, 'forward', 'left', 'right'])
        else :  #state has never been encountered
            Qtable.update({self.state : {None : 0, 'forward' : 0, 'left' : 0 , 'right' : 0}}) #Add state to Qtable dictionary
            action = random.choice([None, 'forward', 'left', 'right'])  #choose one of the actions at random

        # Execute action and get reward
        reward = self.env.act(self, action)  #what was the reward for the chosen action?

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        #Store actions, state and reward as previous_
        state_previous = self.state
        action_previous = action
        reward_previous = reward

        # TODO: Learn policy based on state, action, reward
        if self.steps_counter != 0 :  #make sure it is not the first step in a trial.
            """Bellman equation"""
            Q_hat = Qtable[state_previous][action_previous]
            Q_hat = (1-self.alpha)*Q_hat + (self.alpha * (reward_previous + (self.gamma * max(Qtable[self.state].values()))))
            Qtable[state_previous][action_previous] = Q_hat
            self.Qtable = Qtable

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
