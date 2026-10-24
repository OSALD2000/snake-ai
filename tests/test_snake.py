import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from snake_ai.agent import QNetwork
from snake_ai.game import SnakeGame
from snake_ai.train import run_game


class SnakeGameTests(unittest.TestCase):
    def test_food_and_state_are_valid(self):
        game = SnakeGame(seed=4)
        self.assertEqual(len(game.state()), 11)
        self.assertNotIn(game.food, game.body)
        self.assertEqual(SnakeGame(seed=4).food, game.food)

    def test_apple_grows_snake_and_wall_ends_game(self):
        game = SnakeGame(size=4)
        game.food = (3, 2)
        _, reward, done = game.step(0)
        self.assertEqual((game.score, len(game.body), reward, done), (1, 3, 10.0, False))
        _, reward, done = game.step(0)
        self.assertEqual((reward, done), (-10.0, True))
        with self.assertRaises(ValueError):
            game.step(0)

    def test_moving_into_vacated_tail_cell_is_allowed(self):
        game = SnakeGame(size=4)
        game.body = [(1, 1), (1, 2), (2, 2), (2, 1)]
        game.direction = 0
        game.food = (0, 0)
        _, _, done = game.step(1)
        self.assertFalse(done)
        self.assertEqual(game.body[0], (2, 1))


class QNetworkTests(unittest.TestCase):
    def test_terminal_reward_increases_chosen_q_value(self):
        agent = QNetwork(seed=1)
        state = (0.0,) * 11
        before = agent.values(state)
        agent.learn(state, 1, 5.0, state, done=True)
        after = agent.values(state)
        self.assertGreater(after[1], before[1])
        self.assertEqual(after[0], before[0])
        self.assertEqual(after[2], before[2])

    def test_learning_updates_hidden_weights_and_reduces_error(self):
        agent = QNetwork(seed=1)
        state = (1.0,) * 11
        before = agent.values(state)[0]
        old_weights = [row[:] for row in agent.first]
        agent.learn(state, 0, 2.0, state, done=True)
        self.assertLess(abs(2.0 - agent.values(state)[0]), abs(2.0 - before))
        self.assertNotEqual(agent.first, old_weights)

    def test_saved_model_keeps_predictions(self):
        agent = QNetwork(seed=2)
        state = (1.0,) + (0.0,) * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            agent.save(path)
            self.assertEqual(QNetwork.load(path).values(state), agent.values(state))

    def test_failed_save_preserves_previous_model(self):
        agent = QNetwork(seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text("previous", encoding="utf-8")
            with patch("snake_ai.agent.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    agent.save(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_game_run_is_repeatable(self):
        agent = QNetwork(seed=3)
        self.assertEqual(run_game(agent, seed=5), run_game(agent, seed=5))


if __name__ == "__main__":
    unittest.main()
