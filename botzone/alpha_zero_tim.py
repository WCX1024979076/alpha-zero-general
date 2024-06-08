# -*- coding: utf-8 -*-

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

EPS = 1e-8
PASS = 64
ACTION_SIZE = 65
EMPTY = 0
BLACK = 1
WHITE = -1
CHARS = ['-', 'x', 'o']
DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
       (0, 1), (1, -1), (1, 0), (1, 1))  # 方向向量

INF = 100000
START_TIME = 0
TIME_LIMIT = 3.5

args = {
    'dropout': 0.3,
    'num_channels': 512,
    'BoardSize': 8,
    'ActionSize': 65,
    'cuda': False
}

def is_timeout():
    return time.perf_counter() - START_TIME > TIME_LIMIT

class OthelloNNet(nn.Module):
    def __init__(self, args):
        # game params
        self.board_x = args['BoardSize']
        self.board_y = args['BoardSize']
        self.action_size = args['ActionSize']
        self.args = args

        super(OthelloNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args['num_channels'], 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1)
        self.conv4 = nn.Conv2d(args['num_channels'], args['num_channels'], 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args['num_channels'])
        self.bn2 = nn.BatchNorm2d(args['num_channels'])
        self.bn3 = nn.BatchNorm2d(args['num_channels'])
        self.bn4 = nn.BatchNorm2d(args['num_channels'])

        self.fc1 = nn.Linear(args['num_channels']*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args['num_channels']*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args['dropout'], training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args['dropout'], training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args['cuda']: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.eval()
        with torch.no_grad():
            pi, v = self(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])


# 此处参考：BotZone用户 yyj 的 AlphaZero Bot 源代码，地址https://botzone.org.cn/account/5559e6fc145879554bbf6fbc
class GameBoard:
    def __init__(self):
        board = np.zeros((8, 8), dtype=np.int8)

        board[3, 4] = board[4, 3] = BLACK
        board[3, 3] = board[4, 4] = WHITE

        self.color = BLACK
        self.board = board

    def key(self):
        return self.board * self.color

    def copy(self):
        clone = GameBoard()
        clone.color = self.color
        clone.board = self.board.copy()
        return clone

    def valid_moves(self):
        color = self.color
        board = self.board
        count = 0
        valids = np.zeros((ACTION_SIZE,), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    if self.place(i, j, color, check_only=True):
                        count += 1
                        valids[i * 8 + j] = 1.0
        if count == 0:
            valids[PASS] = 1.0
        return valids

    def place(self, x, y, color, check_only=False):
        board = self.board
        if not check_only:
            board[x][y] = color
        valid = False
        for d in range(8):
            i = x + DIR[d][0]
            j = y + DIR[d][1]
            while 0 <= i and i < 8 and 0 <= j and j < 8 and \
                    board[i][j] == -color:
                i += DIR[d][0]
                j += DIR[d][1]
            if 0 <= i and i < 8 and 0 <= j and j < 8 and \
               board[i][j] == color:
                while True:
                    i -= DIR[d][0]
                    j -= DIR[d][1]
                    if i == x and j == y:
                        break
                    valid = True
                    if check_only:
                        return True
                    board[i][j] = color
        return valid

    def apply_moveXY(self, x, y):
        if x != -1:
            assert self.place(x, y, self.color)
        self.color = -self.color

    def apply_move(self, move):
        if move != PASS:
            # assert self.place(x, y, color), f'Invalid move {xy2pos(x, y)}'
            assert self.place(move // 8, move % 8, self.color)
        self.color = -self.color

    def has_legal_move(self, color):
        board = self.board
        for i in range(8):
            for j in range(8):
                if board[i][j] == 0:
                    if self.place(i, j, color, check_only=True):
                        return True
        return False

    def evaluate(self):
        diff = np.sum(self.board)
        if self.color == WHITE:
            diff = -diff
        if diff > 0:
            return 1
        return -1

    def count(self, color):
        count = 0
        board = self.board
        for i in range(8):
            for j in range(8):
                if board[i][j] == color:
                    count += 1
        return count

    def show(self):
        board = self.board
        print(f'x: {self.count(BLACK)} o: {self.count(WHITE)}')
        print('  ', end='')
        for i in range(8):
            print(f'{i + 1} ', end='')
        print()
        for y in range(8):
            print(f'{chr(ord("A") + y)} ', end='')
            for x in range(8):
                print(f'{CHARS[board[x][y]]} ', end='')
            print()

    def is_game_ended(self):
        return not self.has_legal_move(BLACK) \
            and not self.has_legal_move(WHITE)


class MCTS:

    def __init__(self, nnet, cpuct=1.0):
        self.nnet = nnet
        self.cpuct = cpuct

        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}
        self.Vs = {}        # stores game.getValidMoves for board s

    def best_move(self, game, timeout, temp=1):
        start_time = time.perf_counter()
        count = 0
        while time.perf_counter() - start_time < timeout:
            count += 1
            self.search(game.copy())

        best_action = -1
        best_value = 0
        s = game.key().tostring()
        for a in range(ACTION_SIZE):
            nsa = self.Nsa.get((s, a), 0)
            if nsa > best_value:
                best_value = nsa
                best_action = a
            # debugging
            # if nsa > 0:
            #     print(xy2pos(a % 8, a // 8), nsa)

        return count, best_action

    def search(self, game):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        canonical_board = game.key()
        s = canonical_board.tostring()

        result = self.Es.get(s)
        if result is None:
            result = 0
            if game.is_game_ended():
                result = game.evaluate()
            self.Es[s] = result

        if result != 0:  # 游戏结束
            return -result

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonical_board)
            valids = game.valid_moves()
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                # log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(ACTION_SIZE):
            if not valids[a]:
                continue
            qsa = self.Qsa.get((s, a), None)

            if qsa != None:
                u = qsa + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        game.apply_move(a)
        v = self.search(game)

        qsa = self.Qsa.get((s, a),None)
        if qsa != None:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

# 此处参考：BotZone用户 yyj 的 AlphaZero  Bot 源代码，地址https://botzone.org.cn/account/5559e6fc145879554bbf6fbc
def main():
    global START_TIME

    nnet = OthelloNNet(args)
    nnet.load_checkpoint("data/", "8x8_100checkpoints_best.pth.tar")

    game = GameBoard()
    mcts = MCTS(nnet)

    turn_ID = 0
    while not game.is_game_ended():
        line = input().strip()
        if line == '':
            continue
        full_input = json.loads(line)

        if turn_ID == 0:
            requests = full_input['requests']
            x = requests[0]['x']
            y = requests[0]['y']
            if x >= 0:
                game.apply_moveXY(x, y)
        else:
            x = full_input['x']
            y = full_input['y']
            game.apply_moveXY(x, y)

        debug, move = mcts.best_move(game, TIME_LIMIT, 0)
        if move == PASS:
            x, y = -1, -1
        else:
            x, y = move // 8, move % 8

        print(json.dumps({'response': {'x': x, 'y': y, 'debug': debug}}))
        print('\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n', flush=True)

        game.apply_moveXY(x, y)
        turn_ID += 1


if __name__ == '__main__':
    main()