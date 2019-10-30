import numpy as np
from Wumpus import *

actions = {'norte': np.array([0, 1]), 'sul': np.array([0, -1]), 'leste': np.array([1, 0]), 'oeste': np.array([-1, 0])}
probability = {'frente': [0.7, np.array([[1, 0],[0, 1]])], 'direita': [0.1, np.array([[0, -1],[1, 0]])], 'esquerda': [0.2, np.array([[0, 1],[-1, 0]])]}

world = World(8, 4, actions, probability)

# Wumpus, pit and gold

for wumpus in [(0, 2), (4, 1)]:
    world[wumpus].set_wumpus()

for pit in [(1, 3), (2, 0), (2, 2), (6, 0), (6, 2), (6, 3)]:
    world[pit].set_pit()

for gold in [(1, 2), (5, 1)]:
    world[gold].set_gold()

result = value_iteration(world, 0.9)

for y in range(0, 4):
    y = 3 - y
    for x in range(0, 8):
        print(round(result[x, y].get_utility(),2), end=' ')
    print()

print()

for y in range(0, 4):
    y = 3 - y
    for x in range(0, 8):
        if result[x, y].is_wumpus():
            print('wumpus', end=' ')
        elif result[x, y].is_pit():
            print('pit', end=' ')
        elif result[x, y].is_gold():
            print('gold', end=' ')
        else:
            print(result[x, y].get_policy(), end=' ')
    print()