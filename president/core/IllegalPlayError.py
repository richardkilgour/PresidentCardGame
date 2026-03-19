#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raised when a player returns an illegal play from play().
Carries enough context for GameMaster to act on its policy.
"""


class IllegalPlayError(Exception):
    def __init__(self, player, action, reason: str) -> None:
        self.player = player
        self.action = action
        self.reason = reason
        super().__init__(f"Illegal play by {player.name}: {reason} (action={action})")
