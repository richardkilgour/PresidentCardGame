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


def index_to_meld(index: int) -> Meld:
    # TODO: accept the hand, and play actual cards
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
        self.input = torch.zeros(56 * 3 + 54)
        player_index = self.players.index(player)
        others_indices = [(player_index + i)%4 for i in range(1,4)]
        # Look over the history, populating the input vectors
        for move in self.memory._memory.reverse():
            offset = i * 56
            if len(self.memory._memory) < (4-i):
                # Memory is too short, so default to Noop
                self.input[offset + 55] = 1
                continue
            play = self.memory._memory[i-4]
            if play[0] == self.players[others_indices[i]]:
                # This is the expected player
                other_meld = play[1]
                self.input[offset + meld_to_index(other_meld)] = 1
            else:
                # Noop?
                self.input[offset + 55] = 1
        for i in hand_to_indices(player._hand):
            self.input[56 * 3 + i] = 1
        # Make the training data
        self.target = torch.zeros(55)
        self.target[meld_to_index(meld)] = 1
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
        self.inputs = torch.tensor(data["inputs"], dtype=torch.float32).to(device)
        self.targets = torch.tensor(data["targets"], dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def generate_random_input():
    # Create three zero tensors of size 56
    vec1 = torch.zeros(56, dtype=torch.float32)
    vec2 = torch.zeros(56, dtype=torch.float32)
    vec3 = torch.zeros(56, dtype=torch.float32)

    # Randomly select indices and set them to 1
    vec1[torch.randint(0, 56, (1,))] = 1
    vec2[torch.randint(0, 56, (1,))] = 1
    vec3[torch.randint(0, 56, (1,))] = 1

    # Create a zero tensor of size 54
    vec4 = torch.zeros(54, dtype=torch.float32)

    # Choose between 0 and 14 random indices to set to 1
    num_indices = torch.randint(0, 15, (1,)).item()
    random_indices = torch.randint(0, 54, (num_indices,))
    vec4[random_indices] = 1

    # Concatenate all four vectors into a single input vector
    input_vector = torch.cat([vec1, vec2, vec3, vec4], dim=0)  # (56 * 3 + 54) = (222,)

    return input_vector

def main():
    # Run some tests
    inp, targ = generate_data(1)

    for i, t in zip(inp, targ):
        vec1, vec2, vec3, vec4 = i[:56], i[56:112], i[112:168], i[168:222]
        hand_indices = torch.nonzero(vec4, as_tuple=True)[0].tolist()
        hand = '|'.join([i.__str__() for i in indices_to_hand(hand_indices)])
        print(index_to_meld(torch.argmax(vec1).item()), index_to_meld(torch.argmax(vec2).item()), index_to_meld(torch.argmax(vec3).item()),
              hand, index_to_meld(torch.argmax(t).item()))


if __name__=="__main__":
    main()