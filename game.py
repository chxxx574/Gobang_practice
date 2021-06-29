import numpy as np
import tkinter as tk
import tkinter.messagebox
class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
class GUI_interface(object):
    def __init__(self,width,height,n,player,start_player=1):
        self.board = Board(width=width, height=height, n_in_row=n)
        self.game = Game(self.board)
        self.start_player = start_player
        self.board.init_board(start_player-1)
        self.player = player
        p1, p2 = self.board.players
        player.set_player_ind(p1)
        self.filenum=1
        self.row=self.board.width
        self.column=self.board.height
        self._playboard = tk.Tk()
        self._playboard.title('GOBANG')
        self._playboard.geometry("650x700")
        self.left = tk.Label(self._playboard, text='Player1 with Black', font=('Helvetica', 12))
        self.left.grid(row=0, column=0, sticky=tk.W)
        self._player = tk.Label(self._playboard, text=f'Turn: Player{self.get_turn()}', font=('Helvetica', 12))
        self._player.grid(row=0, column=1, sticky=tk.N)
        self.right = tk.Label(self._playboard, text='Player2 with White', font=('Helvetica', 12))
        self.right.grid(row=0, column=2, sticky=tk.E)

        self._canvas = tk.Canvas(self._playboard, width=800, height=800, background='orange')
        self._canvas.bind('<Button-1>', self._on_canvas_clicked)
        self._canvas.bind('<ButtonRelease-1>', self.play_against)
        self._canvas.bind('<Configure>', self._on_canvas_resized)

        self._canvas.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky=tk.N + tk.S + tk.W + tk.E)

        for i in range(self.row + 1):
            self._canvas.create_line(0, i * (800 / self.row), 800, i * (800 / self.row), fill='gray')
        for j in range(self.column + 1):
            self._canvas.create_line(i * (800 / self.column), 0, i * (800 / self.column), 800, fill='gray')

        self._restart_button = tk.Button(self._playboard,text='Restart',font=('Helvetica', 12),command=self.restart)
        self._restart_button.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        self._title = tk.Label(self._playboard, text='[MCTS_ALPHA_ZERO]   VS   [HUAMN]', font=('Helvetica', 12,'bold'))
        self._title.grid(row=2, column=1, padx=10, pady=10,sticky=tk.S)
        self._exit_button = tk.Button(self._playboard, text='Exit', font=('Helvetica', 12),command=self.destroy)
        self._exit_button.grid(row=2, column=2, padx=10, pady=5, sticky=tk.E)

        self._playboard.rowconfigure(0, weight=0)
        self._playboard.rowconfigure(1, weight=1)
        self._playboard.rowconfigure(2, weight=0)
        self._playboard.columnconfigure(0, weight=1)
        self._playboard.columnconfigure(1, weight=1)
        self._playboard.columnconfigure(2, weight=1)
        current_player = self.board.get_current_player()
        if current_player == 1:
            move = self.player.get_action(self.board)
            self.board.do_move(move)
            self._draw_all()
            self._player['text'] = f'Turn: Player{self.get_turn()}'
    def _on_canvas_clicked(self, event: tk.Event) -> None:
        self._player['text']=f'Turn: Player{self.get_turn()}'
        pixel_width = self._canvas.winfo_width()
        pixel_height = self._canvas.winfo_height()

        nx = int(event.x / (pixel_width / self.column))
        correctx = (nx * 2 + 1) * (pixel_width / self.column / 2.0)
        ny = int(event.y / (pixel_height / self.row))
        correcty = (ny * 2 + 1) * (pixel_height / self.row / 2.0)
        try:
            move = self.board.location_to_move((self.row-1-ny, nx))
            self.board.do_move(move)
            self._draw_all()
            # im = ImageGrab.grab()
            # im.save('./'+str(self.filenum)+'.jpg')
            # im.close()
            # self.filenum=self.filenum+1
        except:
            tk.messagebox.showerror('GOBANG', 'The operation is invalid!')
        end, winner = self.board.game_end()
        if end == True:
            if winner != -1:
                tk.messagebox.showinfo('GOBANG', 'The winner is player' + str(winner))
            else:
                tk.messagebox.showinfo('GOBANG', 'Tie')
            self.destroy()
        else:
            self._player['text'] = f'Turn: Player{self.get_turn()}'
    def play_against(self, event: tk.Event):
        end, winner = self.board.game_end()
        if end==False:
            self._player['text'] = f'Turn: Player{self.get_turn()}'
            move = self.player.get_action(self.board)
            self.board.do_move(move)
            self._draw_all()
            # im = ImageGrab.grab()
            # im.save('./' + str(self.filenum) + '.jpg')
            # im.close()
            # self.filenum = self.filenum + 1
            end, winner = self.board.game_end()
            if end == True:
                if winner != -1:
                    tk.messagebox.showinfo('GOBANG', 'The winner is player' + str(winner))
                else:
                    tk.messagebox.showinfo('GOBANG', 'Tie')
                self.destroy()
            else:
                self._player['text'] = f'Turn: Player{self.get_turn()}'
    def _draw_all(self) -> None:
        self._canvas.delete(tk.ALL)
        pixel_width = self._canvas.winfo_width()
        pixel_height = self._canvas.winfo_height()
        r = self.board.height
        c = self.board.width
        for i in range(c + 1):
            self._canvas.create_line(i * (pixel_width / c), 0, i * (pixel_width / c), pixel_height, fill='gray')
        for j in range(r + 1):
            self._canvas.create_line(0, j * (pixel_height / r), pixel_width, j * (pixel_height / r), fill='gray')
        onex, oney = 1.0 * pixel_width / self.column / 2, 1.0 * pixel_height / self.row / 2
        for i in range(r - 1, -1, -1):
            for j in range(c):
                loc = i * c + j
                p = self.board.states.get(loc, -1)
                if p != -1:
                    ny,nx=self.board.move_to_location(loc)
                    ny=self.row-1-ny
                    x = (nx * 2 + 1) * (pixel_width / self.column / 2.0)
                    y = (ny * 2 + 1) * (pixel_height / self.row / 2.0)
                    topleft_pixel_x, topleft_pixel_y = (x - onex * 0.65, y - oney * 0.65)
                    bottomright_pixel_x, bottomright_pixel_y = (x + onex * 0.65, y + oney * 0.65)
                    if p==1:
                        self._canvas.create_oval(
                            topleft_pixel_x, topleft_pixel_y,
                            bottomright_pixel_x, bottomright_pixel_y,
                            fill='black', outline='white')
                    elif p==2:
                        self._canvas.create_oval(
                            topleft_pixel_x, topleft_pixel_y,
                            bottomright_pixel_x, bottomright_pixel_y,
                            fill='white', outline='black')
    def _on_canvas_resized(self, event: tk.Event) -> None:
        self._draw_all()

    def get_turn(self):
        return self.board.get_current_player()

    def destroy(self):
        self._playboard.destroy()
    def run(self):
        self._playboard.mainloop()
    def restart(self):
        self.board.init_board(self.start_player - 1)
        self._player['text'] = f'Turn: Player{self.get_turn()}'
        current_player = self.board.get_current_player()
        if current_player == 1:
            move = self.player.get_action(self.board)
            self.board.do_move(move)
            self._draw_all()
            self._player['text'] = f'Turn: Player{self.get_turn()}'