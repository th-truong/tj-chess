from network_utils.network_out_interpreter import NetInterpreter


def test_interpreter_round_trip():
    interpreter = NetInterpreter()

    for color in ["white", "black"]:
        interpreter.set_colour_to_play(color)
        for i in range(0, 8):
            for j in range(0, 8):
                for k in range(0, 73):
                    try:
                        move = interpreter.interpret_net_move(i, j, k)
                        assert (i, j, k) == interpreter.interpret_UCI_move(move)
                    except AssertionError:
                        pass
