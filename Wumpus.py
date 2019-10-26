import numpy as np


# Cell class. Also that represents a state
class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.wumpus = False
        self.pit = False
        self.gold = False
        self.utility = 0
        self.return_value = 0
        self.policy = None

    def is_wumpus(self):
        return self.wumpus

    def is_pit(self):
        return self.pit

    def is_gold(self):
        return self.gold

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
        self.set_return(-100)

    def set_pit(self):
        self.pit = True
        self.set_return(-50)

    def set_gold(self):
        self.gold = True
        self.set_return(100)

    def set_return(self, return_value):
        self.return_value = return_value

    def set_utility(self, utility):
        self.utility = utility

    def set_policy(self, action):
        self.policy = action


# World
class World:
    def __init__(self, x_size, y_size, actions, probability):
        self.states = []
        for y in range(0, y_size):
            for x in range(0, x_size):
                self.states.append(Cell(x, y))
        self.actions = actions
        self.probability = probability
        self.x_size = x_size
        self.y_size = y_size

    def __getitem__(self, tup):
        x_index, y_index = tup
        #print('get item: ', tup)
        #print(y_index * self.x_size + x_index)
        return self.states[y_index * self.x_size + x_index]

    def get_actions(self):
        return self.actions

    def get_action(self, action):
        return self.actions.get(action)

    def get_probability(self):
        return self.probability.items()

    def get_dimensions(self):
        return [self.x_size, self.y_size]

    def get_next_state_utility(self, direction, state):
        next_index = state.get_index() + direction
        # Board hit
        if next_index[0] < 0 or next_index[0] >= self.x_size or next_index[1] < 0 or next_index[1] >= self.y_size:
            return state.get_utility() - 1
        # Common case
        else:
            #print('get_next_state_utility: ', next_index)
            next_state = self[next_index[0], next_index[1]]
            return next_state.get_utility() - 0.1


# Value iteration
def value_iteration(problem, gama):

    # Auxiliary functions
    def get_action_expected_utility(state, action):
        expected_utility = 0
        for key, value in problem.get_probability():
            # Get final direction from desired action and movement possibly performed
            direction = value[1].dot(problem.get_action(action))
            u = problem.get_next_state_utility(direction, state)
            expected_utility += value[0] * u
        return expected_utility

    def chose_best_action(x, y):
        actions = list(problem.get_actions().keys())
        state = problem[x, y]
        best_action = actions[0]
        best_utility = get_action_expected_utility(state, best_action)
        for action in actions[1:]:
            u = get_action_expected_utility(state, action)
            if u > best_utility:
                best_action = action
                best_utility = u
        return best_action, best_utility

    def update_utility(x, y):
        state = problem[x, y]
        # chose best action
        _, best_utility = chose_best_action(x, y)
        # r(s)
        reinforcement = state.get_return()
        return reinforcement + gama * best_utility

    # Calculate utilities
    for i in range(1000):
        for x in range(problem.get_dimensions()[0]):
            for y in range(problem.get_dimensions()[1]):
                problem[x, y].set_utility(update_utility(x, y))

    # Calculate policy
    for x in range(problem.get_dimensions()[0]):
        for y in range(problem.get_dimensions()[1]):
            best_action, _ = chose_best_action(x, y)
            problem[x, y].set_policy(best_action)

    return problem
