"""
pip install numpy
pip install chess [https://python-chess.readthedocs.io/en/latest/]
pip install matplotlib
"""

import numpy as np
import chess
import random
import datetime
import matplotlib.pyplot as plt

EPISODES = 50000

class Node:
    def __init__(self, state, parent):
        self.state = state
        self.children = []
        self.parent = parent
        self.S = 0
        self.N = 0
        self.wins = 0
        self.nwins = []

    def __str__(self, level=0):
        """ Print the tree """
        if self.N != 0:
            ret = "\t"*level+repr((self.wins/self.N)*100)+"\n"
        else:
            ret = "\t"*level+repr(0)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def init_children(self):
        """ Vul de children variabel aan met de juiste child nodes"""
        children = []
        legal_moves = list(chess.Board(self.state).legal_moves)
        for move in legal_moves:
            temp_board = chess.Board(self.state)
            temp_board.push_san(str(move))
            children.append(Node(temp_board.fen(), self))
        self.children = children

    def backprop_update(self, reward):
        # We Updaten de state vanwaar we de rollout hebben gedaan
        self.N += 1
        self.S += reward
        if reward == 1:
            self.wins += 1
        self.nwins.append((self.wins/self.N)*100)
        # We Updaten al de nodes die we bezocht hebben voor de rollout
        parent = self.parent
        curr_reward = reward
        while parent != None: # de rootnode heeft geen parent
            curr_reward *= -1
            parent.N += 1
            parent.S += curr_reward
            if curr_reward == 1:
                parent.wins += 1
            parent.nwins.append((parent.wins/parent.N)*100)
            parent = parent.parent

class MonteCarloTreeSearch:
    def __init__(self, C):
        self.root = Node(chess.Board().fen(), None)
        self.C = C

    def UCB1(self, node):
        # Delen door 0 vermijden
        if node.N == 0:
            return np.inf
        winrate = node.S / node.N
        return winrate + self.C * np.sqrt(np.log(self.root.N) / node.N)



    #picks the child with the highest UCB1
    def find_promising_node(self, node):
        if node.children == []:
            node.init_children()

        values = {}
        for child in node.children:
            values[child] = self.UCB1(child)

        return max(values, key=values.get)

    def select(self, node):
        promising_node = self.find_promising_node(node)
        '''
          Als de node nog niet bezocht is weten we automatisch
          dat het nog geen kinderen heeft.

          We zoeken hier door de boom tot we een node vinden
          die nog niet bezocht is met een hoge UCB1 score
        '''
        if promising_node.N == 0:
            return promising_node
        else:
            while promising_node.N != 0:
                promising_node = self.find_promising_node(promising_node)
            return promising_node

    def rollout(self, state):
        # Het schaakbord vanuit een bepaalde state
        board = chess.Board(state)
        # We houden de huidige speler bij om zo de winnaar te bepalen
        player = board.turn

        #simulate 1 random game
        while board.is_game_over() == False:
            play = str(random.choice(list(board.legal_moves)))
            board.push_san(play)
            winner = not board.turn

        if board.is_checkmate() and player == winner:
            return 1
        elif board.is_stalemate():
            return 0
        return -1

    def train(self, i): # episodes per stap
        self.simulate(i)

    def simulate(self, iterations, node=None):
        # Als we geen node als root node ingeven nemen we de rootnode van het spel.
        if node == None:
            node = self.root
            state = node.state

        for i in range(iterations):
            # De node die geselecteerd is volgens zijn UCB1 waarde
            promising_node = self.select(node)
            # De reward van een random rollout vanuit deze node
            reward = self.rollout(promising_node.state)
            # Alle waarden updaten na een rollout
            promising_node.backprop_update(reward)

MCTS = MonteCarloTreeSearch(2)
MCTS.train(EPISODES)


wins = MCTS.root.nwins
x = range(len(wins))
y = []
plt.plot(x, wins)
plt.ylabel('Winpercentage')
plt.xlabel('Episodes')
plt.show()
