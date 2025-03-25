import torch
from torch.utils.data import Dataset

from asshole.core.CardGameListener import CardGameListener
from asshole.core.GameMaster import GameMaster
from asshole.core.Meld import Meld
from asshole.core.PlayingCard import PlayingCard

# TODO: Find a better place for these constants
PASS_INDEX = 54
NOOP_INDEX = 55  # AKA waiting
MAX_HAND_SIZE = 14

def meld_to_index(meld: Meld):
    # Index is based on the card values, plus the number of cards played
    # 0 is a single 3, 1 is a double 3, 2 is a triple 3 and 3 is a quadruple 3
    # 4 is a single 4, 5 is a double 4 , and so forth.
    # None is waiting
    if not meld:
        return NOOP_INDEX
    # If the meld is a number, it's the position of the player (0=King etc)
    if meld in [0, 1, 2, 3]:
        # TODO: The models can't differentiate winning hand with pass
        return PASS_INDEX
    if meld.cards:
        return 4 * meld.cards[0].get_value() + len(meld.cards) - 1
    return PASS_INDEX


def index_to_meld(index: int) -> Meld | None:
    # TODO: accept the hand, and return actual cards
    if index == PASS_INDEX:
        # "Pass" is an empty meld
        return Meld()
    elif index == NOOP_INDEX:
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
        # This will add the current play to the memory
        super().notify_play(player, meld)

        assert self.memory._memory[-1].player == player
        assert self.memory._memory[-1].meld == meld

        # Add the player's hand to the input
        hand = hand_to_indices(player._hand)

        # Get up to 3 previous plays
        prev_plays = []
        # Skip the current play
        for _, play, desc in self.memory.previous_plays_generator(1):
            cards = meld_to_index(play)
            prev_plays.insert(0, cards)
            if len(prev_plays) >= 3:
                break

        # Fill with "noop" if we don't have enough plays
        while len(prev_plays) < 3:
            prev_plays.append(NOOP_INDEX)  # Noop code

        # Make the training data
        padding = [PASS_INDEX] * (MAX_HAND_SIZE-len(hand))
        self.input = prev_plays + hand + padding
        self.target = meld_to_index(meld)
    def get_data(self):
        input_ = self.input
        target = self.target
        self.input = None
        self.target = None
        return input_, target


def generate_data(number_of_rounds = 100):
    from asshole.players.PlayerSplitter import PlayerSplitter
    gm = GameMaster()
    data_grabber = DataGrabber()
    gm.add_listener(data_grabber)
    # Learn to play like a simple player
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
    idxs = torch.randint(0, NOOP_INDEX+1, (batch_size, 3))

    # Generate hands with different lengths for each sample
    hands = []
    for _ in range(batch_size):
        num_indices = torch.randint(0, 15, (1,)).item()  # Between 0 and 14 values
        hand = torch.randint(0, PASS_INDEX, (num_indices,))  # Values in {0, ..., 53}
        padding = torch.full((MAX_HAND_SIZE - num_indices,), PASS_INDEX, dtype=torch.long)  # Pad with 55
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
        hand = filter(lambda val: val<PASS_INDEX, i[3:])
        hand_str = '|'.join([i.__str__() for i in indices_to_hand(hand)])
        print(index_to_meld(plays[0]), index_to_meld(plays[1]), index_to_meld(plays[2]),
              hand_str, index_to_meld(t))


if __name__=="__main__":
    main()