# -*- coding: utf-8 -*-
from __future__ import print_function
from game import GUI_interface
from mcts_alphaZero import MCTSPlayer
from policy import PolicyValueNet
from config import Conf


def run():
    policy_param = Conf.model_file
    best_policy = PolicyValueNet(policy_param)
    mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                             c_puct=Conf.n_in_row,
                             n_playout=Conf.n_playout)  # set larger n_playout for better performance
    g = GUI_interface(Conf.board_width, Conf.board_height, Conf.n_in_row, mcts_player)
    g.run()


if __name__ == '__main__':
    run()
