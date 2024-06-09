from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .HexLogic import Board
import numpy as np

class HexGame(Game):
    CX = [-1, 0, 1, 0, -1, 1]
    CY = [0, -1, 0, 1, 1, -1]

    square_content = {
        -1: "X", # Block
        +0: "-",
        +1: "O" # White
    }

    def inBoard(self, x, y) :
        return x >= 0 and y >=0 and x < self.n and y < self.n
    
    def isWinHelper(self, board, visited, startX, startY, color):
        if not self.inBoard(startX, startY) :
            return False
        loc = startX * self.n + startY
        if board[startX][startY] != color :
           return False
        if visited[loc]:
            return False
        visited[loc] = True
        if color == -1 and startX == self.n - 1 :
            return True
        if color == 1 and startY == self.n - 1 :
            return True
        for i in range(6):
            if self.isWinHelper(board, visited, startX + self.CX[i], startY + self.CY[i], color) :
                return True
        return False

    def checkWinner(self, board) :
        visited = [False] * (self.n * self.n)
        for y in range(self.n) :
            if board[0][y] == -1 :
                if self.isWinHelper(board, visited, 0, y, -1) :
                    return -1
        for i in range(self.n * self.n) :
            visited[i] = False
        for x in range(self.n) :
            if board[x][0] == 1 :
                if self.isWinHelper(board, visited, x, 0, 1) :
                    return 1
        return 0

    @staticmethod
    def getSquarePiece(piece):
        return HexGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        if player == -1 : # 黑棋下的棋子进行反转
            move = (action%self.n, int(action/self.n))
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves()
        if len(legalMoves)==0:
            print("error: no legalMoves\n");
            exit(-1)
            # valids[-1]=1
            # return np.array(valids)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        result = self.checkWinner(b.pieces)
        if result != 0 :
            return result
        if b.has_legal_moves():
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return board
        else :
            return (player*board).T

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2)  # 1 for pass
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            if y%2 == 0 :
                print(" ")
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(HexGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
