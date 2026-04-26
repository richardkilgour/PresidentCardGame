#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerConsole is a human-controlled player with console input/output.

The play() function presents possible melds and waits for user input.
Typing 'q' at any prompt saves a checkpoint and exits cleanly.
"""
from __future__ import annotations

import logging
import sys

from president.core.AbstractPlayer import AbstractPlayer
from president.core.GameRecord import GameRecord
from president.core.Meld import Meld


class PlayerConsole(AbstractPlayer):

    def __init__(self, name):
        super().__init__(name)
        self._record: GameRecord | None = None

    def set_record(self, record: GameRecord) -> None:
        """Attach the GameRecord so the player can save and quit mid-game."""
        self._record = record

    # -------------------------------------------------------------------------
    # Listener callbacks
    # -------------------------------------------------------------------------

    def surrender_cards(self, cards, receiver):
        print(f'  ► You must give {", ".join(str(c) for c in cards)} to {receiver.name}')
        super().surrender_cards(cards, receiver)

    def award_cards(self, cards, giver):
        print(f'  ◄ You received {", ".join(str(c) for c in cards)} from {giver.name}')
        super().award_cards(cards, giver)

    def notify_hand_won(self, winner):
        super().notify_hand_won(winner)
        print(f'\n{"─" * 40}')
        print(f'  {winner.name} wins the hand.')
        print(f'{"─" * 40}')

    def notify_play(self, player, meld):
        super().notify_play(player, meld)
        remaining = player.report_remaining_cards() - len(meld)
        if player != self:
            print(f'  {player.name} plays {meld}  ({remaining} cards left)')
        self._update_status(player, meld)

    def notify_pass(self, player):
        super().notify_pass(player)
        if player != self:
            print(f'  {player.name} passes.')
            self._update_status(player, Meld())

    def notify_waiting(self, player):
        super().notify_waiting(player)
        if player != self:
            self._update_status(player, None)

    def notify_played_out(self, player, pos):
        super().notify_played_out(player, pos)
        rank_name = self.ranking_names[pos]
        if player == self:
            print(f'\n  ★ You ({player.name}) finished as {rank_name}!')
        else:
            print(f'  {player.name} finished as {rank_name}.')
            self._update_status(player, rank_name)

    def notify_episode_end(self, final_ranks, starting_ranks):
        if input("Save this game to your history? [y/n] ").lower() == 'y':
            super().notify_episode_end(final_ranks, starting_ranks)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def _update_status(self, player, status) -> None:
        """Update the display status for a player if they are in our player list."""
        if player in self.players:
            self.player_status[self.players.index(player)] = status

    def _show_table(self) -> None:
        """Print the current table state from this player's perspective."""
        print(f'\n{"═" * 40}')
        for i, player in enumerate(self.players):
            if player is None:
                continue
            you_str = "  ◄ you" if player == self else ""
            status = self.player_status[i]
            if status is None:
                status_str = "waiting"
            elif isinstance(status, Meld):
                status_str = f'played {status}' if status.cards else 'passed'
            elif isinstance(status, str):
                status_str = status
            else:
                status_str = str(status)
            print(f'  {player.name:15} {player.report_remaining_cards():2} cards   {status_str}{you_str}')
        print(f'{"─" * 40}')
        print(f'  Your hand: {self}')
        print(f'{"═" * 40}\n')

    # -------------------------------------------------------------------------
    # Play
    # -------------------------------------------------------------------------

    def play(self, valid_plays) -> Meld:
        """
        Present possible melds and prompt the user to select one.
        Returns a Meld, or '␆' as a no-op if input is not ready.
        Type 'q' to save a checkpoint and quit.
        """
        self._show_table()

        options = valid_plays

        print("Select a play:")
        for i, meld in enumerate(options):
            split_str = "  (split)" if meld.cards and self.will_split(meld) else ""
            print(f'  [{i}] {meld}{split_str}')
        print("  [q] Save and quit")

        user_input = input("\nYour choice: ").strip().lower()

        if user_input in ('q', 'quit'):
            self._save_and_quit()

        if not user_input:
            logging.info(f'{self.name} selects default option 0: {options[0]}')
            return options[0]

        try:
            card_index = int(user_input)
            if card_index < 0 or card_index >= len(options):
                print(f'  Please enter a number from 0 to {len(options) - 1}.')
                return '␆'
        except ValueError:
            print(f'  Please enter a number from 0 to {len(options) - 1}, or q to quit.')
            return '␆'

        logging.info(f'{self.name} plays option {card_index}: {options[card_index]}')
        return options[card_index]

    def _save_and_quit(self) -> None:
        """Save the game record and exit cleanly."""
        if self._record:
            path = GameRecord.stamped_path("quit_save")
            self._record.save(path)
            print(f'\n  Game saved to {path}. Resume with: --restore {path}')
        else:
            print('\n  No record configured — progress will be lost.')
        sys.exit(0)
