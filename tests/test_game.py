import numpy as np
import pytest

from game.connect4 import ConnectFour


def play_sequence(actions):
    game = ConnectFour()
    for action in actions:
        if game.is_terminal():
            break
        game.step(action)
    return game


def test_initial_state():
    game = ConnectFour()
    assert game.board.shape == (6, 7)
    assert (game.board == 0).all()
    assert game.get_current_player() == 1
    assert not game.is_terminal()
    assert game.get_winner() == 0
    assert np.array_equal(game.get_legal_actions(), np.arange(7))
    assert game.move_count == 0


def test_observation_shape_and_perspective():
    game = ConnectFour()
    obs = game.get_observation()
    assert obs.shape == (2, 6, 7)
    assert obs.dtype == np.float32
    assert obs.sum() == 0
    game.step(3)
    obs = game.get_observation()
    assert game.get_current_player() == 2
    assert obs[0].sum() == 0
    assert obs[1, 5, 3] == 1.0
    assert obs[0, 5, 3] == 0.0


def test_piece_gravity_and_alternation():
    game = ConnectFour()
    game.step(0)
    game.step(0)
    assert game.board[5, 0] == 1
    assert game.board[4, 0] == 2
    assert game.get_current_player() == 1
    assert 0 in game.get_legal_actions()


def test_full_column_is_illegal():
    game = ConnectFour()
    for _ in range(6):
        game.step(1)
    assert not game.is_legal(1)
    assert 1 not in game.get_legal_actions()
    with pytest.raises(ValueError):
        game.step(1)


def test_invalid_actions():
    game = ConnectFour()
    with pytest.raises(ValueError):
        game.step(-1)
    with pytest.raises(ValueError):
        game.step(7)
    with pytest.raises(TypeError):
        game.step(1.2)


def test_horizontal_win():
    game = play_sequence([0, 6, 1, 6, 2, 6, 3])
    assert game.is_terminal()
    assert game.get_winner() == 1
    assert game.get_result(1) == 1.0
    assert game.get_result(2) == -1.0


def test_vertical_win():
    game = play_sequence([0, 1, 0, 1, 0, 1, 0])
    assert game.is_terminal()
    assert game.get_winner() == 1
    assert game.last_action == 0


def test_diagonal_down_right_win():
    game = play_sequence([0, 1, 1, 2, 2, 6, 2, 6, 3, 3, 3, 4, 3])
    assert game.get_winner() == 1
    assert game.is_terminal()


def test_diagonal_down_left_win():
    game = play_sequence([3, 6, 2, 2, 3, 1, 0, 4, 5, 0, 2, 2, 1, 0, 2, 3, 3])
    assert game.get_winner() == 1


def test_draw_detection():
    board = np.array(
        [
            [2, 2, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 1, 1, 2],
            [2, 2, 1, 2, 2, 1, 1],
            [1, 1, 2, 2, 1, 2, 2],
            [1, 1, 2, 1, 1, 2, 1],
            [2, 1, 2, 1, 2, 2, 1],
        ],
        dtype=np.int8,
    )
    game = ConnectFour(board=board, current_player=1)
    assert game.is_terminal()
    assert game.get_winner() == 0
    assert game.get_result(1) == 0.0
    assert game.get_result(2) == 0.0
    assert len(game.get_legal_actions()) == 0


def test_terminal_step_raises():
    game = play_sequence([0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(RuntimeError):
        game.step(2)


def test_clone_is_independent():
    game = ConnectFour()
    game.step(3)
    clone = game.clone()
    clone.step(4)
    assert game.move_count == 1
    assert clone.move_count == 2
    assert game.board[5, 4] == 0
    assert clone.board[5, 4] == 2


def test_reset_clears_everything():
    game = play_sequence([0, 1, 0, 1, 0, 1, 0])
    game.reset()
    assert (game.board == 0).all()
    assert not game.is_terminal()
    assert game.get_winner() == 0
    assert game.get_current_player() == 1
    assert game.last_action is None


def test_invalid_board_rejected():
    with pytest.raises(ValueError):
        ConnectFour(board=np.zeros((5, 7), dtype=np.int8))
    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1
    board[5, 3] = 2
    with pytest.raises(ValueError):
        ConnectFour(board=board)


def test_both_players_can_win():
    # Player two wins horizontally on the next move.
    game = play_sequence([0, 1, 6, 1, 5, 1, 4, 1])
    assert game.get_winner() == 2


def test_get_result_before_terminal_raises():
    game = ConnectFour()
    with pytest.raises(RuntimeError):
        game.get_result(1)
