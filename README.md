## gobang based on a simplified AlphaZero model
これを金子先生に読んでもらえれば、とても嬉しいです。急いで書いたので、バグがあるかもしれませんので、ご容赦ください。

This is a simplified version of a gobang game done with reference to AlphaZero, the board is designed with reference to the author @Junxiao Song, and the framework is based on keras.I also made a graphical interface, so it's more convenient to play Gobang!

### Requirements
- 3.6 <= Python <= 3.7
- Tensorflow == 1.15.0
- keras == 2.0.5

### Getting Started
To play with provided models, run the following script from the directory:  
```
python play.py  
```
You may modify config.py to try different settlement.

To train the AI model directly run:   
```
python train.py
```
The models (best_policy.model and current_policy.model) will be saved every a few updates (default 100).  


### Details of the model
Since Gobang is much simpler than Go, it can be simplified by many ways.
Firstly, in AlphaGo Zero, the input is first passed through 20 or 40 convolution-based residual blocks, and then connected to a 2 or 3 layer network to respectively get the strategy and value output, and the whole network has more than 40 or 80 layers, which is updated very slowly. The model I used for Gobang greatly simplifies this network structure, starting with a common 3-layer full convolutional network, using 32, 64 and 128 filters respectively, and using the ReLu activation function. In the part of policy, 4 filters are used for dimensionality reduction, followed by a fully-connected layer that directly outputs the probability of each position on the board using softmax function; In the part of value,  a fully-connected layer with 64 neurons is added, and finally a fully-connected layer with a tanh nonlinear function directly outputs the position scores between them. The depth of the whole strategy value network is only 5~6 layers, and the training and prediction are relatively fast.

AlphaGo Zero uses a total of 17 19*19 binary feature planes to describe the current position, where the first 16 planes describe the position of both players corresponding to the last 8 moves, and the last plane describes the color of the current player's pieces, while Gobang has been tested not to require so many recent moves, so I choose to take the nearest move, which use 2 planes to describe the position of the previous move. Since our next move is often near our opponent's previous move, the addition of the third plane is a good indicator for the strategy network to determine which positions should have a higher probability of being played, and the fourth plane describes the sequence of moves in Gobang.

The input of the strategy value network is the current situation description, and the output is the probability of each feasible action in the current situation and the score of the current situation, and the data we collect during the self-play process is used to train the strategy value network. According to the above, the goal of our training is to make the probability of action output of the strategy value network closer to the probability of MCTS output, so that the position score output of the strategy value network can more accurately predict the real game outcome. From an optimization perspective, we are continuously minimizing the loss function on the self-play dataset: ![image](https://github.com/chxxx574/Gobang_practice/blob/main/img_e.png), where the third term is a regular term used to prevent overfitting


### Deficiencies and parts that can be optimized later
1.The model performs well on a 8x8 board and poorly on a 15x15 board even after 3000 rounds of training.I think the main reason is that the size of the board determines the number of past moves to be taken into account, and since this simple model only uses data from the most recent move, it does not perform well on larger boards, increasing the number of rounds using history should improve this problem.

2.In the training process, especially the process of self-play is too slow, because the neural network layer is relatively simple, does not constitute a bottleneck in speed, after analysis found that only one core of the cpu is used, the subsequent optimization can be multi-core simultaneous work to increase the training speed.

3.In some cases, using the gpu version of tensorflow will cause the gpu to fail to load and cause the program to freeze. The reason may be related to the high python version, please use the tensorflow version without gpu and Python version under 3.7.
