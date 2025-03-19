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
        # "Pass" is an empty meld
        return Meld()
    elif index == 55:
        # Waiting is None
        return None
    meld = Meld(PlayingCard(index))
    while index % 4:
        index -= 1
        meld = Meld(PlayingCard(index), meld)
    return meld

def indices_to_hand(indices: list[int]):
    hand = []
    for i in indices:
        meld = index_to_meld(i)
        if meld:
            if meld.cards:
                hand.append(meld.cards[-1])
            else:
                # "Pass"
                hand.append(meld)
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

        previous_player_index = (self.players.index(player) + 3) % 4
        # Move previous player to the front
        players = self.players[previous_player_index:] + self.players[:previous_player_index]

        # Get up to 3 previous plays
        prev_plays = []
        for play in self.memory.previous_plays_generator(players):
            cards = meld_to_index(play)
            prev_plays.append(cards)
            if len(prev_plays) >= 3:
                break

        # Fill with "noop" if we don't have enough plays
        while len(prev_plays) < 3:
            prev_plays.append(55)  # Noop code

        # Reverse to match original order
        prev_plays = list(reversed(prev_plays))

        # Make the training data
        padding = [54] * (14-len(hand))
        self.input = prev_plays + hand + padding
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
        self.inputs = torch.tensor(data["inputs"], dtype=torch.int64).to(device)
        self.targets = torch.tensor(data["targets"], dtype=torch.int64).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def generate_random_input(batch_size):
    # Randomly select indices for previous plays (batch_size x 3 values in {0, ..., 55})
    idxs = torch.randint(0, 56, (batch_size, 3))

    # Generate hands with different lengths for each sample
    hands = []
    for _ in range(batch_size):
        num_indices = torch.randint(0, 15, (1,)).item()  # Between 0 and 14 values
        hand = torch.randint(0, 54, (num_indices,))  # Values in {0, ..., 53}
        padding = torch.full((14 - num_indices,), 54, dtype=torch.long)  # Pad with 55
        hands.append(torch.cat([hand, padding]))

    # Stack hands into a batch (batch_size x 14)
    hands = torch.stack(hands)

    # Concatenate idxs and hands to form the final batch tensor (batch_size x 17)
    return torch.cat([idxs, hands], dim=1)

def main():
    # Run some tests
    inp, targ = generate_data(1)

    for i, t in zip(inp, targ):
        plays = i[:3]
        hand = filter(lambda val: val<54, i[3:])
        hand_str = '|'.join([i.__str__() for i in indices_to_hand(hand)])
        print(index_to_meld(plays[0]), index_to_meld(plays[1]), index_to_meld(plays[2]),
              hand_str, index_to_meld(t))


if __name__=="__main__":
    main()