#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MOVED — this module now lives at president.training.data.StateEncoder.
This stub re-exports everything for backwards compatibility.
Update your imports to:
    from president.training.data.StateEncoder import StateEncoder, ...
"""
from president.training.data.StateEncoder import (  # noqa: F401
    StateEncoder,
    STATE_BITS, HAND_SIZE_BITS, MELD_BITS, HAND_BITS,
    BLOCK_0_SIZE, BLOCK_N_SIZE, TOTAL_BITS,
    IDX_WAITING, IDX_HAS_PLAYED, IDX_WON_ROUND,
    MAX_REGULAR_VALUE, JOKER_OFFSET,
)
