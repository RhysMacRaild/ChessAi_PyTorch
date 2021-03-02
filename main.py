import chess
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessAI:

    @staticmethod
    def get_board_as_array(board):
        piece_dict = board.piece_map()
        array = np.zeros(64, dtype=int)
        for key in piece_dict:
            is_white = piece_dict[key].color
            piece_number = piece_dict[key].piece_type

            if is_white:
                unique_num = piece_number + 6
            else:
                unique_num = piece_number

            array[key] = unique_num
        return array

    @staticmethod
    def convert_to_one_hot(array):
        one_hot = np.zeros((array.size, array.max()+1))
        one_hot[np.arange(array.size), array] = 1
        return one_hot.astype(np.float32).flatten()

    def generate_all_moves(self, board):
        possible_moves = []
        for i in board.generate_legal_moves():
            board = chess.Board()
            board.push_uci(i.uci())
            possible_board = self.get_board_as_array(board)
            one_hot = self.convert_to_one_hot(possible_board)
            possible_moves.append([one_hot, i.uci()])
        return possible_moves

    def rate_all_moves(self, possible_moves):
        for index in range(len(possible_moves)):
            possible_moves[index][0] = self.rate_move(possible_moves[index][0])
        return possible_moves

    def rate_move(self, possible_move):
        ...
        return 0  # placeholder

#https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(832, 1200)
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.softmax(self.fc1(x), dim=0)
        x = F.softmax(self.fc2(x), dim=0)
        x = F.softmax(self.fc3(x), dim=0)
        x = self.fc4(x)
        return x


def main():
    net = Net()
    print(net)

    board = chess.Board()
    ai = ChessAI()
    array = ai.generate_all_moves(board)
    # array = ai.rate_all_moves(array)
    input = torch.from_numpy(array[0][0])
    out = net(input)
    print(out)

    #
    # start = time.time()
    #
    # array = ai.generate_all_moves(board)
    # array = ai.rate_all_moves(array)
    #
    # end = time.time()
    # print(end - start)


if __name__ == "__main__":
    main()
