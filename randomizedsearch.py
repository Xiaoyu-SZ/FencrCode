# coding=utf-8

import argparse
import sys

from happyrec.utilities.commandline import *
from happyrec.tasks.RandomizedSearch import RandomizedSearch


def main():
    grid_parser = argparse.ArgumentParser(description='RandomizedSearch Args')
    grid_parser = RandomizedSearch.add_task_args(grid_parser)
    grid_args, _ = grid_parser.parse_known_args()
    gridsearch = RandomizedSearch(**vars(grid_args))
    gridsearch.run()
    return


if __name__ == '__main__':
    main()
