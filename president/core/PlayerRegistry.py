#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PlayerRegistry is a factory for creating AbstractPlayer instances.

Responsibilities:
  - Register player types with optional constructor kwargs
  - Create player instances by name
  - Provide the full list of registered entries for evaluation

Used by:
  - GameMaster:         create players by type name
  - GameCheckpoint:     restore players from saved type names
  - EvaluationRunner:   create fresh instances per game
  - MatchupGenerator:   enumerate all registered types
"""
from dataclasses import dataclass, field
from typing import Any

from president.core.AbstractPlayer import AbstractPlayer


@dataclass
class PlayerEntry:
    """A registered player type with its display name and constructor kwargs."""
    name: str                       # Display name e.g. "Naive", "RL_v2"
    player_type: type               # The class to instantiate
    kwargs: dict[str, Any] = field(default_factory=dict)  # e.g. {"model_path": "v2.pt"}

    def __str__(self):
        return self.name


class PlayerRegistry:
    """
    Factory and registry for AbstractPlayer subclasses.

    Example usage:
        registry = PlayerRegistry()
        registry.register(PlayerNaive,    name="Naive")
        registry.register(PlayerSimple,   name="Simple")
        registry.register(PlayerRL,       name="RL_v2", model_path="models/v2.pt")

        player = registry.create("Naive", player_name="Alice")
        entries = registry.all_entries()
    """

    def __init__(self) -> None:
        self._entries: dict[str, PlayerEntry] = {}

    @classmethod
    def from_config(cls, config: dict) -> "PlayerRegistry":
        """
        Build a PlayerRegistry by dynamically importing player types from config.
        Assumes player class names follow the convention Player{type_name}.

        Args:
            config: Parsed config.yaml as a dict.

        Returns:
            A populated PlayerRegistry.
        """
        import importlib
        registry = cls()
        seen = set()
        for key in ['player1', 'player2', 'player3', 'player4']:
            p = config[key]
            if p.get('console', False):
                continue
            type_name = p['type']
            if type_name in seen:
                continue
            module = importlib.import_module(f'president.players.Player{type_name}')
            player_class = getattr(module, f'Player{type_name}')
            kwargs = {k: v for k, v in p.items()
                      if k not in ('name', 'type', 'type_', 'console')}
            registry.register(player_class, name=type_name, **kwargs)
            seen.add(type_name)
        return registry

    def register(self, player_type: type[AbstractPlayer],
                 name: str = None, **kwargs) -> None:
        """
        Register a player type.

        Args:
            player_type: The AbstractPlayer subclass to register.
            name:        Display name. Defaults to the class name.
            **kwargs:    Additional constructor arguments (e.g. model_path).

        Raises:
            ValueError: If the name is already registered.
        """
        if name is None:
            name = player_type.__name__
        if name in self._entries:
            raise ValueError(
                f"Player type '{name}' is already registered. "
                f"Use a unique name for each entry."
            )
        if not issubclass(player_type, AbstractPlayer):
            raise TypeError(
                f"{player_type.__name__} must be a subclass of AbstractPlayer."
            )
        self._entries[name] = PlayerEntry(
            name=name,
            player_type=player_type,
            kwargs=kwargs,
        )

    def create(self, name: str, player_name: str = None) -> AbstractPlayer:
        """
        Create a new player instance by registered name.

        Args:
            name:        The registered display name e.g. "Naive", "RL_v2".
            player_name: The name to give the player instance.
                         Defaults to the registered display name.

        Returns:
            A fresh AbstractPlayer instance.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._entries:
            raise KeyError(
                f"Player type '{name}' is not registered. "
                f"Available types: {list(self._entries.keys())}"
            )
        entry = self._entries[name]
        player_name = player_name or entry.name
        return entry.player_type(player_name, **entry.kwargs)

    def all_entries(self) -> list[PlayerEntry]:
        """Return all registered entries in registration order."""
        return list(self._entries.values())

    def names(self) -> list[str]:
        """Return all registered display names."""
        return list(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __str__(self) -> str:
        return f"PlayerRegistry({', '.join(self._entries.keys())})"