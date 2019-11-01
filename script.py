import numpy as np
from Wumpus import *

# Acoes sao dicionarios cujas chaves sao os nomes das acoes e valores sao os vetores de deslocamento
actions = {'norte': np.array([0, 1]), 'sul': np.array([0, -1]), 'leste': np.array([1, 0]), 'oeste': np.array([-1, 0])}
# Probabilidades sao dicionarios cujas chaves sao as possiveis direcoes, e os valores sao vetores com o valor da
# probabilidade para a direcao, e com uma matriz de rotacao que permite obter a direcao final ao multiplicar
# o vetor da acao desejada pela matriz de rotacao da direcao
probability = {'frente': [0.7, np.array([[1, 0],[0, 1]])], 'direita': [0.1, np.array([[0, -1],[1, 0]])], 'esquerda': [0.2, np.array([[0, 1],[-1, 0]])]}

# Instanciacao do problema
world = World(8, 4, actions, probability)

# Posicionamento dos Wumpus, pit e ouro
for wumpus in [(0, 2), (4, 1)]:
    world[wumpus].set_wumpus()

for pit in [(1, 3), (2, 0), (2, 2), (6, 0), (6, 2), (6, 3)]:
    world[pit].set_pit()

for gold in [(1, 2), (5, 1)]:
    world[gold].set_gold()

# Obtercao do resultado
result = value_iteration(world, 0.9)

# Impressao simplificada dos resultados
print('\nUtilidades dos estados\n')
for y in range(0, 4):
    y = 3 - y
    for x in range(0, 8):
        print(round(result[x, y].get_utility(),2), end=' ')
    print()

print('\nPolitica obtida\n')

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
