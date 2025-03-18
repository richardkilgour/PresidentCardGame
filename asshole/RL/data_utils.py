import torch
from torch.utils.data import Dataset

from asshole.RL.file_utils import load_expert_data
from asshole.core.CardGameListener import CardGameListener
from asshole.core.GameMaster import GameMaster
from asshole.core.Meld import Meld
from asshole.core.PlayingCard import PlayingCard
from asshole.players.PlayerSplitter import PlayerSplitter


def meld_to_index(meld: Meld):
    # Index is based on the card values, plus the number of cards played
    # 0 is a single 3, 1 is a double 3, 2 is a triple 3 and 3 is a quadruple 3
    # 4 is a single 4, 5 is a double 4 , and so forth.
    # index 54 is a pass
    if meld.cards:
        return 4 * meld.cards[0].get_value() + len(meld.cards) - 1
    return 54


def index_to_meld(index: int) -> Meld | None:
    # TODO: accept the hand, and return actual cards
    if index == 54:
        return Meld()
    elif index == 55:
        return None
    meld = Meld(PlayingCard(index))
    while index % 4:
        index -= 1
        meld = Meld(PlayingCard(index), meld)
    return meld

def indices_to_hand(indices: list[int]):
    hand = []
    for i in indices:
        hand.append(index_to_meld(i).cards[-1])
    return hand


def hand_to_indices(hand) -> list[int]:
    # We don't care about the suit of the cards, just the number of them
    indices = []
    meld = Meld()
    for card in hand:
        if not meld.cards or card.get_value() != meld.cards[0].get_value():
            # Add single meld
            meld = Meld(card)
            indices.append(meld_to_index(meld))
        else:
            # Duplicate card
            meld = Meld(card, meld)
            indices.append(meld_to_index(meld))
    return indices


class DataGrabber(CardGameListener):
    def __init__(self):
        super().__init__()
        self.input = None
        self.target = None
    def notify_play(self, player, meld):
        super().notify_play(player, meld)
        # Add the hand to the input
        hand = hand_to_indices(player._hand)

        prev_plays = [-1, -1, -1]
        previous_player_index = (self.players.index(player) + 3) % 4
        # Move previous player to the front
        players = self.players[previous_player_index:] + self.players[:previous_player_index]
        # Look over the history, populating the input vectors
        offset = 2
        # Ignore the most recent play - it's covered by the meld
        for move in reversed(self.memory._memory[:-1]):
            # Find the player that made this move (move is [player, meld])
            while move[0] != players[0]:
                # Not the expected player - Assume they passed
                prev_plays[offset] = 54
                offset -= 1
                # Bring the last player to the front
                players = [players[-1]] + players[:-1]
                if offset < 0:
                    break
            if move[0] == players[0]:
                # This is the expected player
                other_meld = move[1]
                prev_plays[offset] = meld_to_index(other_meld)
                offset -= 1
                # Bring the last player to the front
                players = [players[-1]] + players[:-1]
            if offset < 0:
                break
        while offset >= 0:
            # Assume noop
            prev_plays[offset] = 55
            offset -= 1

        # Make the training data
        self.input = prev_plays + [hand]
        self.target = meld_to_index(meld)
    def get_data(self):
        input_ = self.input
        target = self.target
        self.input = None
        self.target = None
        return input_, target


def generate_data(number_of_rounds = 100):
    gm = GameMaster()
    data_grabber = DataGrabber()
    gm.add_listener(data_grabber)
    gm.add_player(PlayerSplitter(f'name_1'))
    gm.add_player(PlayerSplitter(f'name_2'))
    gm.add_player(PlayerSplitter(f'name_3'))
    gm.add_player(PlayerSplitter(f'name_4'))
    gm.start(number_of_rounds)
    inputs = []
    targets = []
    while not gm.step():
        input_, target = data_grabber.get_data()
        if input_ is None:
            continue
        inputs.append(input_)
        targets.append(target)
    return inputs, targets


class ExpertDataset(Dataset):
    def __init__(self, device, filename="expert_data.pt"):
        data = torch.load(filename)
        self.inputs = data["inputs"]
        self.targets = data["targets"]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        print(self.inputs[idx], self.targets[idx])
        return self.inputs[idx], self.targets[idx]


def generate_random_input():
    # Randomly select indices for previous plays
    idx1 = torch.randint(0, 56, (1,))
    idx2 = torch.randint(0, 56, (1,))
    idx3 = torch.randint(0, 56, (1,))

    # Choose between 0 and 14 random indices
    num_indices = torch.randint(0, 15, (1,)).item()
    hand = torch.randint(0, 54, (num_indices,))

    return idx1, idx2, idx3, hand

def main():
    # Run some tests
    inp, targ = generate_data(1)

    for i, t in zip(inp, targ):
        vec1, vec2, vec3, hand = i
        hand = '|'.join([i.__str__() for i in indices_to_hand(hand)])
        print(index_to_meld(vec1), index_to_meld(vec2), index_to_meld(vec3),
              hand, index_to_meld(t))


if __name__=="__main__":
    main()