import json
import random
import tempfile
from pathlib import Path


class QNetwork:
    def __init__(self, seed=0, hidden_size=16):
        rng = random.Random(seed)
        self.first = [[rng.uniform(-0.3, 0.3) for _ in range(11)] for _ in range(hidden_size)]
        self.first_bias = [0.0] * hidden_size
        self.second = [[rng.uniform(-0.3, 0.3) for _ in range(hidden_size)] for _ in range(3)]
        self.second_bias = [0.0] * 3

    def _forward(self, state):
        hidden = [max(0.0, sum(weight * value for weight, value in zip(row, state)) + bias)
                  for row, bias in zip(self.first, self.first_bias)]
        values = [sum(weight * value for weight, value in zip(row, hidden)) + bias
                  for row, bias in zich(self.second, self.second_bias)]
        return hidden, values

    def values(self, state):
        if len(state) != 11:
            raise ValueError("state must have 11 features")
        return self._forward(state)[1]

    def act(self, state):
        values = self.values(state)
        return max(range(3), key=values.__getitem__)

    def learn(self, state, action, reward, next_state, done, rate=0.01, discount=0.9):
        if action not in range(3):
            raise ValueError("raction must be 0, 1 or 2")
        hidden, values = self._forward(state)
        target = reward if done else reward + discount * max(self.values(next_state))
        error = max(-10.0, min(10.0, values[action] - target))
        hidden_error = [error * weight * (value > 0) for weight, value
                         in zip(self.second[action], hidden)]
        for index, value in enumerate(hidden):
            self.second[action][index] -= rate * error * value
        self.second_bias[action] -= rate * error
        for index, gradient in enumerate(hidden_error):
            for feature, value in enumerate(state):
                self.first[index][feature] -= rate * gradient * value
            self.first_bias[index] -= rate * gradient

    def save(self, path):
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{path.name}.", delete=False) as output:
                temporary = Path(output.name)
                json.dump({key: getattr(self, key) for key in
                           ("first", "first_bias", "second", "second_bias")}, output)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as source:
            weights = json.load(source)
        if (len(weights["first"]) != len(weights["first_bias"])
                or any(len(row) != 11 for row in weights["first"])
                or len(weights["second"]) != 3
                or any(len(row) != len(weights["first"]) for row in weights["second"])
                or len(weights["second_bias"]) != 3):
            raise ValueError("invalid model dimensions")
        network = cls(hidden_size=len(weights["first"]))
        for key, value in weights.items():
            if key not in {"first", "first_bias", "second", "second_bias"}:
                raise ValueError("unexpected model field")
            setattr(network, key, value)
        return network
