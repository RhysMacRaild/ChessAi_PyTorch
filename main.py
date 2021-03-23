import chess
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import random
import torch.optim as optim
import operator

dtype = torch.cuda.FloatTensor
device = torch.device('cuda')


class Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.ratings = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx][0], self.samples[idx][1], idx

    def append(self, sample):
        self.samples.append(sample)

    def set_rating(self, index, rating, move, tensor):
        self.ratings[index] = (rating, move, tensor)

    def get_ratings(self):
        return self.ratings

    #Not sure why this is needed but it works
    def process(self):
        for sample, truth, index in self:
            self.samples[index][0] = sample[0]


class ChessAI:
    def __init__(self, network):
        self.save_probability = 0.5
        self.network = network
        self.network.to(device)
        self.board_states = []

    @staticmethod
    # Return the chess board an an int array
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
    # Convert an array to one hot encoded
    def convert_to_one_hot(array):
        one_hot = np.zeros((array.size, array.max() + 1))
        one_hot[np.arange(array.size), array] = 1
        return one_hot.astype(np.float32).flatten()

    # Generate all legal moves
    def generate_all_moves(self, board):
        possible_moves = []
        for i in board.generate_legal_moves():
            board_tmp = copy.copy(board)
            board_tmp.push_uci(i.uci())
            possible_board = self.get_board_as_array(board_tmp)
            one_hot = self.convert_to_one_hot(possible_board)
            one_hot = torch.from_numpy(one_hot).cuda()
            possible_moves.append([one_hot, i.uci()])
            # possible_moves.append([one_hot, i.uci()])
        return Dataset(possible_moves)

    # Rate the probability of all possible moves resulting in a win
    def rate_all_moves(self, possible_moves):
        batch_size = 10
        workers = 1
        for tensor, move, index in DataLoader(possible_moves):  # for in datathigno instead?
            possible_moves.set_rating(index.item(), self.rate_move_tensor(tensor), move[0], tensor)
        return possible_moves

    # # Rate the probability of all possible moves resulting in a win
    # def rate_all_moves(self, possible_moves):
    #     for index in range(len(possible_moves)):  # for in datathigno instead?
    #         possible_moves[index][0] = self.rate_move_tensor(possible_moves[index])
    #     possible_moves.sort(key=self.sortWin)
    #     return possible_moves

    # Calculate the probability of this move resulting in a winning game
    def rate_move_numpy(self, possible_move):
        input = torch.from_numpy(possible_move).type(dtype)
        out = self.network(input)
        return out

    def rate_move_tensor(self, possible_move):
        out = self.network(possible_move)
        return out

    @staticmethod
    # Sort key
    def sortWin(elem):
        tensor = elem[1][0]
        return tensor.data[0][0].item()

    def save_board_state(self, move):
        # [Predicted Probability, Result]
        board_state = [move, None]
        self.board_states.append(board_state)

    # Pick the move with the highest percentage change of winning
    def calculate_next_move(self, board):
        possible_moves = self.generate_all_moves(board)
        rated_moves = self.rate_all_moves(possible_moves)
        # Whites Move
        if board.turn:
            move = max(rated_moves.ratings.items(), key=self.sortWin)[1]
            # move = rated_moves[0][1]

        # Blacks move
        else:
            move = min(rated_moves.ratings.items(), key=self.sortWin)[1]
            # move = rated_moves[-1][1]

        if random.random() < self.save_probability:
            self.save_board_state(move[2])

        return move

    # Pick the next move at random
    def random_next_move(self, board):
        possible_moves = self.generate_all_moves(board)
        rated_moves = self.rate_all_moves(possible_moves)
        index = random.randint(0, len(rated_moves) - 1)
        move = rated_moves.ratings[index]

        if random.random() < self.save_probability:
            self.save_board_state(move[2])

        return move

    def set_result(self, result):
        if result == "1/2-1/2":
            result_tensor = torch.from_numpy(np.array([0.5, 0.5], np.float32)).type(dtype)
        elif result == "1-0":
            result_tensor = torch.from_numpy(np.array([1, 0], np.float32)).type(dtype)
        elif result == "0-1":
            result_tensor = torch.from_numpy(np.array([0, 1], np.float32)).type(dtype)
        else:
            print("ERROR SETTING RESULT\n")

        for i in self.board_states:
            i[1] = result_tensor

        return


# https://medium.com/dair-ai/a-simple-neural-network-from-scratch-with-pytorch-and-google-colab-c7f3830618e0
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(832, 1200)
        self.fc2 = nn.Linear(1200, 840)
        self.fc3 = nn.Linear(840, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.fc3(x), dim=0)
        x = F.softmax(self.out(x), dim=1)

        # x = F.softmax(x, dim=0)
        return x

def main():
    training_loop_count = 0
    while True:
        training_loop_count += 1
        net = Net()
        net.load_state_dict(torch.load("model" + str(training_loop_count) + ".pt"))

        # Generate training data
        training_set = []
        for i in range(5000):
            board = chess.Board()
            ai = ChessAI(net)
            while not board.is_game_over():
                if random.random() < 1/10:
                    next_move = ai.random_next_move(board)
                else:
                    next_move = ai.calculate_next_move(board)
                board.push_san(next_move[1])
                # print(board)
                # print("\n")
            # exit()
            print(board.result(), i)
            ai.set_result(board.result())
            if board.result() == "1/2-1/2":
                if random.random() < 1/10:
                    training_set += ai.board_states[:]
            else:
                training_set += ai.board_states[:]

        # Training Loop
        dataset = Dataset(training_set)
        dataset.process()
        data_loader = DataLoader(dataset, shuffle=True, drop_last=True, batch_size=128)
        criterion = nn.MSELoss()
        optimiser = optim.SGD(net.parameters(), lr=0.08)
        num_epochs = 20000
        for epoch in range(num_epochs):
            if epoch % 250 == 0:
                torch.save(net.state_dict(), "model" + str(training_loop_count) + ".pt")
                print("Saving Model")

            for input, truth, index in data_loader:
                output = ai.rate_move_tensor(input)
                optimiser.zero_grad()
                loss = criterion(output, truth)
                loss.backward()
                optimiser.step()
            print(loss, epoch)
        torch.save(net.state_dict(), "model" + str(training_loop_count) + ".pt")


if __name__ == "__main__":
    main()
