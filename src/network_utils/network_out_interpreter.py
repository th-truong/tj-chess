"""[summary]
"""


class NetInterpreter():

    def __init__(self):
        self.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.rows = ['1', '2', '3', '4', '5', '6', '7', '8']

        self.moves_list = self._move_mapping()

    def interpret_net_move(self, column, row, move):
        starting_position = self.columns[column] + self.rows[row]

        offsets = self.moves_list[move]

        if (column + offsets[1] < 0 or row + offsets[0] < 0 or
            column + offsets[1] > 7 or row + offsets[0] > 7):  # for out of bounds moves
            final_position = "INVALID"
        elif 64 <= move <= 72 and self.rows[row] != '7':  # for illegal promotions
            final_position = "INVALID"
        else:
            final_position = self.columns[column + offsets[1]] + self.rows[row + offsets[0]]
            if 64 <= move <= 66:  # these are the rook promotions
                final_position += "r"
            elif 67 <= move <= 69:  # knight promos
                final_position += "n"
            elif 70 <= move <= 72:  # bishop promos
                final_position += "b"

        return starting_position + final_position


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
        pawn_underpromos.append((1,0))  # N move to rook
        pawn_underpromos.append((1,1))  # NE move to rook
        pawn_underpromos.append((1,-1))  # NW move to rook
        pawn_underpromos.append((1,0))  # N move to knight
        pawn_underpromos.append((1,1))  # NE move to knight
        pawn_underpromos.append((1,-1))  # NW move to knight
        pawn_underpromos.append((1,0))  # N move to bishop
        pawn_underpromos.append((1,1))  # NE move to bishop
        pawn_underpromos.append((1,-1))  # NW move to bishop

        all_moves = queen_moves + knight_moves + pawn_underpromos

        return all_moves
