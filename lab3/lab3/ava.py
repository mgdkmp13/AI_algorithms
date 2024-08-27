from exceptions import GameplayException
from connect4 import Connect4
from lab3.alphabetaagent import AlphaBetaAgent
from randomagent import RandomAgent
from minmaxagent import MinMaxAgent
from minmaxagent_h import MinMaxAgentH
from alphabetaagent import AlphaBetaAgent

connect4 = Connect4(width=7, height=6, )
#agent1 = RandomAgent('o')
#agent2 = RandomAgent('x')
#agent1 = AlphaBetaAgent('o')
agent2 = AlphaBetaAgent('o')
agent1 = MinMaxAgent('x')
#agent2 = MinMaxAgent('x')
#agent1 = MinMaxAgentH('o')
#agent2 = MinMaxAgentH('x')
while not connect4.game_over:
    connect4.draw()
    try:
        if connect4.who_moves == agent1.my_token:
            n_column = agent1.decide(connect4, 4)
        else:
            n_column = agent2.decide(connect4, 4)
        connect4.drop_token(n_column)
    except (ValueError, GameplayException):
        print('invalid move')

connect4.draw()
