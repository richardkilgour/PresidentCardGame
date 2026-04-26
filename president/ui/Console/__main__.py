#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console UI entry point.

Usage:
    python -m president.ui.Console                                   # offline, new game
    python -m president.ui.Console --restore save.json               # resume offline game
    python -m president.ui.Console --display save.json               # inspect save, no run
    python -m president.ui.Console --server localhost:5000           # join server (anonymous)
    python -m president.ui.Console --server 192.168.1.5:5000 \\
                                   --username alice --password s3cr3t
"""
from __future__ import annotations

import argparse

from president.core.GameRecord import GameRecord


def main():
    parser = argparse.ArgumentParser(description='Play PresidentCardGame (console).')
    parser.add_argument('--restore', type=str, metavar='FILE',
                        help='Resume a game from a saved file (offline only).')
    parser.add_argument('--display', type=str, metavar='FILE',
                        help='Inspect a saved file and exit.')
    parser.add_argument('--server', type=str, metavar='HOST[:PORT]',
                        help='Server address for online play (e.g. localhost:5000).')
    parser.add_argument('--username', type=str, help='Username for online play.')
    parser.add_argument('--password', type=str, help='Password for online play.')
    parser.add_argument('--name', type=str, help='Display name for anonymous online play.')
    args = parser.parse_args()

    if args.display:
        print(GameRecord.display(args.display))
        return

    if args.server:
        from president.ui.Console.online import run_online
        run_online(args.server, args.username, args.password, args.name)
        return

    from president.ui.Console.offline import run_offline
    run_offline(args)


if __name__ == '__main__':
    main()
