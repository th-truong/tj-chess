from network_utils.network_out_interpreter import NetInterpreter


def test_interpreter_round_trip():
    interpreter = NetInterpreter()

    interpreter.set_colour_to_play("white")
    print("Testing White, errors:")
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 73):
                move = interpreter.interpret_net_move(i, j, k)
                # TODO: we should explicity test for invalid moves
                if move == 'INVALID':
                    continue
                assert (i, j, k) == interpreter.interpret_UCI_move(move)

    interpreter.set_colour_to_play("black")
    print("Testing Black, errors:")
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 73):
                move = interpreter.interpret_net_move(i, j, k)
                if move == 'INVALID':
                    continue
                assert (i, j, k) == interpreter.interpret_UCI_move(move)
