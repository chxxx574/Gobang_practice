from collections import deque
class Conf:
    board_width = 8
    board_height = 8
    n_in_row = 5
    learn_rate = 2e-3
    l2_const = 1e-4
    lr_multiplier = 1.0
    temp = 1.0
    n_playout = 4000
    episode_len=0
    c_puct = 5
    buffer_size = 10000
    batch_size = 512  # mini-batch size for training
    data_buffer = deque(maxlen=buffer_size)
    play_batch_size = 1
    epochs = 5  # num of train_steps for each update
    kl_targ = 0.02
    check_freq = 100
    game_batch_num = 3000
    best_win_ratio = 0.0
    pure_mcts_playout_num = 1000
    init_model=None # start training from an initial policy-value net or not
    model_file= 'best_policy_8.model'