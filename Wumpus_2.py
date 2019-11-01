import copy

import numpy as np


class Cell:
    # Cell class. Also represents a state.
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.wumpus = False
        self.pit = False
        self.gold = False
        self.empty = True
        self.utility = 0
        self.return_value = 0
        self.policy = None

    def is_wumpus(self):
        return self.wumpus

    def is_pit(self):
        return self.pit

    def is_gold(self):
        return self.gold

    def is_empty(self):
        return self.empty

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_index(self):
        return np.array([self.x, self.y])

    def get_return(self):
        return self.return_value

    def get_utility(self):
        return self.utility

    def get_policy(self):
        return self.policy

    def set_wumpus(self):
        self.wumpus = True
        self.empty = False
        self.set_return(-100)

    def set_pit(self):
        self.pit = True
        self.empty = False
        self.set_return(-50)

    def set_gold(self):
        self.gold = True
        self.empty = False
        self.set_return(100)

    def set_return(self, return_value):
        self.return_value = return_value

    def set_utility(self, utility):
        self.utility = utility

    def set_policy(self, action):
        self.policy = action


class World:
    # This class represents the world of Wumpus
    def __init__(self, x_size, y_size, actions, probability):
        self.states = []
        for y in range(0, y_size):
            for x in range(0, x_size):
                self.states.append(Cell(x, y))
        self.new_states = copy.deepcopy(self.states)
        self.actions = actions
        self.probability = probability
        self.x_size = x_size
        self.y_size = y_size

    def __getitem__(self, tup):
        x_index, y_index = tup
        return self.states[y_index * self.x_size + x_index]

    # Next methods returns the MDP parts
    def get_actions(self):
        return self.actions

    def get_action(self, action):
        return self.actions.get(action)

    def get_probability(self):
        return self.probability.items()

    def get_dimensions(self):
        return [self.x_size, self.y_size]

    def get_next_state_utility(self, direction, state):
        # Given a state and an action(deterministic), this methods returns the utility of the target state and a
        # boolean if is a board hit
        next_index = state.get_index() + direction
        # Board hit
        if self.is_board_action(direction, state):
            return state.get_utility(), True
        # Common case
        else:
            next_state = self[next_index[0], next_index[1]]
            return next_state.get_utility(), False

    def get_mean_utility(self, reinforcement, gama):
        # Returns the mean utility of all states. It is used to calculate te expected utility of states not empty
        # because next action is a random movement uniformly distributed along all possible states
        mean = 0
        cost = 0.1
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
                mean += reinforcement - cost + gama * self[x, y].get_utility()
        return mean/(self.y_size*self.x_size)

    def set_utility(self, x_index, y_index, u):
        # Changes the utility of a state to a calculated value given as parameter
        index = y_index * self.x_size + x_index
        self.new_states[index].set_utility(u)

    def update_states(self):
        # Used to update all states utility after an iteration cycle
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
                index = y * self.x_size + x
                self.states[index].set_utility(self.new_states[index].get_utility())

    def get_delta(self):
        # Used to calculate the biggest difference between actual and next states, in order to check the convergence
        # of the value iteration algorithm
        delta = 0
        for y in range(0, self.y_size):
            for x in range(0, self.x_size):
                index = y * self.x_size + x
                delta = abs(self.states[index].get_utility() - self.new_states[index].get_utility())
        return delta

    def is_board_action(self, direction, state):
        # This method is used to verify if a movement causes a board hit
        next_index = state.get_index() + direction
        # Board hit
        if next_index[0] < 0 or next_index[0] >= self.x_size or next_index[1] < 0 or next_index[1] >= self.y_size:
            return True
        # Common case
        else:
            next_state = self[next_index[0], next_index[1]]
            return False


# Value iteration
def value_iteration(problem, gama):
    # This function implements the iteration value algorithm

    # Auxiliary functions
    def get_action_expected_total_utility(state, action):
        # returns the expected utility of a action from a state
        expected_total_utility = 0
        reinforcement = state.get_return()
        for key, value in problem.get_probability():
            # Get final direction from desired action and movement possibly performed
            direction = value[1].dot(problem.get_action(action))
            utility, board_hit = problem.get_next_state_utility(direction, state)
            if board_hit:
                cost = 1
            else:
                cost = 0.1
            total_utility = reinforcement - cost + gama*utility
            expected_total_utility += value[0] * total_utility
        return expected_total_utility

    def chose_best_action(x, y):
        # This function returns the best action between all possibilities, based upon their expected utility
        actions = list(problem.get_actions().keys())
        state = problem[x, y]
        best_action = actions[0]
        max_expected_total_utility = get_action_expected_total_utility(state, best_action)
        for action in actions[1:]:
            expected_total_utility = get_action_expected_total_utility(state, action)
            if expected_total_utility > max_expected_total_utility:
                best_action = action
                max_expected_total_utility = expected_total_utility
        return best_action, max_expected_total_utility

    def chose_policy(x, y):
        # Returns the policy for a state, after the end of iteration value algorithm, which is the action with
        # the best expected utility
        return chose_best_action(x, y)

    def update_utility(x, y):
        # This function gets the best action for a state, calculates its reinforcement and returns the updated utility
        # Is used in value iteration algorithm to get new values of utility for each state along each iteration
        state = problem[x, y]
        # chose best action
        if state.is_empty():
            _, max_expected_total_utility = chose_best_action(x, y)
        else:
            reinforcement = state.get_return()
            cost = 0.1
            max_expected_total_utility = problem.get_mean_utility(reinforcement, gama)
        return max_expected_total_utility

    # Calculate utilities: value iteration algorithm
    delta = 1
    while delta > 0.001:
        for x in range(problem.get_dimensions()[0]):
            for y in range(problem.get_dimensions()[1]):
                problem.set_utility(x, y, update_utility(x, y))
        delta = problem.get_delta()
        problem.update_states()

    # Calculate policy
    for x in range(problem.get_dimensions()[0]):
        for y in range(problem.get_dimensions()[1]):
            best_action, _ = chose_policy(x, y)
            problem[x, y].set_policy(best_action)

    return problem
