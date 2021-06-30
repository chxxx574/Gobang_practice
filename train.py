from __future__ import print_function
import random
import numpy as np
from collections import defaultdict
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy import PolicyValueNet
from config import Conf

board = Board(width=Conf.board_width,
              height=Conf.board_height,
              n_in_row=Conf.n_in_row)
game = Game(board)

if Conf.init_model:
    policy_value_net = PolicyValueNet(Conf.init_model)
else:
    policy_value_net = PolicyValueNet()
mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=Conf.c_puct, n_playout=Conf.n_playout, is_selfplay=1)


def get_equi_data(play_data):
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(Conf.board_height, Conf.board_width)), i)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data


def collect_selfplay_data(n_games=1):
    for i in range(n_games):
        winner, play_data = game.start_self_play(mcts_player, temp=Conf.temp)
        play_data = list(play_data)[:]
        Conf.episode_len = len(play_data)
        # augment the data
        play_data = get_equi_data(play_data)
        Conf.data_buffer.extend(play_data)


def policy_update():
    mini_batch = random.sample(Conf.data_buffer, Conf.batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = policy_value_net.policy_value(state_batch)
    for i in range(Conf.epochs):
        loss, entropy = policy_value_net.train_step(
            state_batch,
            mcts_probs_batch,
            winner_batch,
            Conf.learn_rate * Conf.lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1)
                     )
        if kl > Conf.kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    if kl > Conf.kl_targ * 2 and Conf.lr_multiplier > 0.1:
        Conf.lr_multiplier /= 1.5
    elif kl < Conf.kl_targ / 2 and Conf.lr_multiplier < 10:
        Conf.lr_multiplier *= 1.5

    explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) / np.var(np.array(winner_batch)))
    explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) / np.var(np.array(winner_batch)))
    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{},"
           "explained_var_old:{:.3f},"
           "explained_var_new:{:.3f}"
           ).format(kl,
                    Conf.lr_multiplier,
                    loss,
                    entropy,
                    explained_var_old,
                    explained_var_new))
    return loss, entropy


def policy_evaluate(n_games=10):
    current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                     c_puct=Conf.c_puct,
                                     n_playout=Conf.n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5,
                                 n_playout=Conf.pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    for i in range(n_games):
        winner = game.start_play(current_mcts_player,
                                 pure_mcts_player,
                                 start_player=i % 2,
                                 is_shown=0)
        win_cnt[winner] += 1
    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
        Conf.pure_mcts_playout_num,
        win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


def run():
    try:
        for i in range(Conf.game_batch_num):
            collect_selfplay_data(Conf.play_batch_size)
            print("batch i:{}, episode_len:{}".format(
                i + 1, Conf.episode_len))
            if len(Conf.data_buffer) > Conf.batch_size:
                loss, entropy = policy_update()
            # check the performance of the current model,
            # and save the model params
            if (i + 1) % Conf.check_freq == 0:
                print("current self-play batch: {}".format(i + 1))
                win_ratio = policy_evaluate()
                policy_value_net.save_model('./current_policy_' + str(Conf.board_width) + '.model')
                if win_ratio > Conf.best_win_ratio:
                    print("New best policy!!!!!!!!")
                    best_win_ratio = win_ratio
                    # update the best_policy
                    policy_value_net.save_model('./best_policy_' + str(Conf.board_width) + '.model')
                    if (best_win_ratio == 1.0 and
                            Conf.pure_mcts_playout_num < 5000):
                        Conf.pure_mcts_playout_num += 1000
                        Conf.best_win_ratio = 0.0
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
