"""[summary]
"""


class NetInterpreter():

    def __init__(self):
        """Class to interpret the tj-chess network outputs. Used to convert from UCI notation to tj-chess output notation.

            self.columns represent the a-h columns on the board, self.rows represents the 1-8 rows.
            The perspective of the board is determined by self.colour. Defaults to white.
            The columns and rows are reversed if self.colour == "black". This represents the player
            to move as always being in the "south" position of the board.
        """
        self.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.rows = ['1', '2', '3', '4', '5', '6', '7', '8']
        self.colour = "white"

        self.moves_list = self._move_mapping()

    def set_colour_to_play(self, new_colour):
        """Sets the colour of the object so that the orientation of the board has the player
        to move as the south position of the board.

        Args:
            new_colour ([str]): [String representing the player to move, valid values are
            "white" and "black]

        Raises:
            ValueError: [new_colour is not "white" or "black"]
        """
        if new_colour != "white" and new_colour != "black":
            raise ValueError("new_colour must be either \"white\" or \"black\".")
        if new_colour != self.colour:
            self.columns.reverse()
            self.rows.reverse()
            self.colour = new_colour

    def interpret_net_move(self, column, row, move):
        """Given the argmax(tj-chess output), which provides the column, row, and move indices,
        output the UCI notation, if it is a valid move.

        Args:
            column ([int]): [column index of the location of the maximum value in the output of the tj-chess network.]
            row ([int]): [row index of the location of the maximum value in the output of the tj-chess network.]
            move ([int]): [move index of the location of the maximum value in the output of the tj-chess network.]

        Returns:
            [str]: [description]
        """
        starting_position = self.columns[column] + self.rows[row]

        offsets = self.moves_list[move]

        if (column + offsets[1] < 0 or row + offsets[0] < 0 or column + offsets[1] > 7 or row + offsets[0] > 7):  # for out of bounds moves
            move = "INVALID"
        elif 64 <= move <= 72 and self.rows[row] != self.rows[-2]:  # for illegal promotions
            move = "INVALID"
        else:
            final_position = self.columns[column + offsets[1]] + self.rows[row + offsets[0]]
            if 64 <= move <= 66:  # these are the rook promotions
                final_position += "r"
            elif 67 <= move <= 69:  # knight promos
                final_position += "n"
            elif 70 <= move <= 72:  # bishop promos
                final_position += "b"
            move = starting_position + final_position

        return move

    def interpret_UCI_move(self, UCI_move):
        """Given the UCI move, outputs the column, row, and move index which corresponds to the same
        move as output by tj-chess

        Args:
            UCI_move ([str]): [Must be a valid UCI move]

        Raises:
            TypeError: [UCI_move must be a str.]

        Returns:
            [int, int, int]: [Contains the column, row, and move_index corresponding to the input UCI_move]
        """
        if type(UCI_move) is not str:
            raise TypeError(f"UCI_move must be a str, not {type(UCI_move)}")
        # maybe add some value checking later

        start_column = self.columns.index(UCI_move[0])
        start_row = self.rows.index(UCI_move[1])

        end_column = self.columns.index(UCI_move[2])
        end_row = self.rows.index(UCI_move[3])

        offsets = (end_row - start_row, end_column - start_column)
        move_index = self.moves_list.index(offsets)
        if len(UCI_move) == 5:
            promotion = UCI_move[4]
            if promotion == "r":  # these are the rook promotions
                move_index = self.moves_list[64:67].index(offsets) + 64
            elif promotion == "n":  # knight promos
                move_index = self.moves_list[67:70].index(offsets) + 67
            elif promotion == "b":  # bishop promos
                move_index = self.moves_list[70:73].index(offsets) + 70

        return start_column, start_row, move_index

    def _move_mapping(self):
        """Provide the mapping of moves needed to the 73 possible moves for any given square.
           Each tuple is the required offset for the move, given as (row_offset, column_offset).

           The descriptions are of the directions
           North, North-East, East, South-East, South, South-West, West, North-West

        """
        # queen moves, corresponds to planes 0->55 out of 73 planes encoding moves
        queen_moves = []
        for i in range(1, 8):
            queen_moves.append((i, 0))  # North moves
            queen_moves.append((i, i))  # North-East moves
            queen_moves.append((0, i))  # East moves
            queen_moves.append((-i, i))  # South-East moves
            queen_moves.append((-i, 0))  # South moves
            queen_moves.append((-i, -i))  # South-West moves
            queen_moves.append((0, -i))  # West moves
            queen_moves.append((i, -i))  # North-West moves

        # knight moves, corresponds to planes 56-> 63
        knight_moves = []
        knight_moves.append((2, 1))
        knight_moves.append((1, 2))
        knight_moves.append((-1, 2))
        knight_moves.append((-2, 1))
        knight_moves.append((-2, -1))
        knight_moves.append((-1, -2))
        knight_moves.append((1, -2))
        knight_moves.append((2, -1))

        # pawn underpromotions, we will default to a queen promotion so we need to encode
        # forward + left diagonal + right diagonal underpromotions to knights, rooks, and bishops
        # an example UCI notation for this is "g7g8x"
        # where x is q for queen, n for knight, r for rook, and b for bishop
        # this includes the final 64->72 planes
        pawn_underpromos = []
        pawn_underpromos.append((1, 0))  # N move to rook
        pawn_underpromos.append((1, 1))  # NE move to rook
        pawn_underpromos.append((1, -1))  # NW move to rook
        pawn_underpromos.append((1, 0))  # N move to knight
        pawn_underpromos.append((1, 1))  # NE move to knight
        pawn_underpromos.append((1, -1))  # NW move to knight
        pawn_underpromos.append((1, 0))  # N move to bishop
        pawn_underpromos.append((1, 1))  # NE move to bishop
        pawn_underpromos.append((1, -1))  # NW move to bishop

        all_moves = queen_moves + knight_moves + pawn_underpromos

        return all_moves
