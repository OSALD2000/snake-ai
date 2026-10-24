# Snake AI

[Deutsch](README.de.md)

A small, reproducible Snake experiment with a neural Q-learning agent. It is a
clean successor to my older `snake_ki` project. The game, network, and tests were
rebuilt for this repository; the old X11 controller, compiled game, and model files
are not included.

## Run

Python 3.10+ is the only requirement. No API key, display server, or package
installation is needed.

```sh
python3 -m snake_ai.train --episodes 3000 --seed 0
python3 -m unittest discover -s tests -v
```

Training saves `model.json` locally. It is ignored by Git. Use `--model PATH` to
choose another output file, `--episodes N` to change training length, and
`--eval-games N` to change the evaluation sample.

## How it works

- An 8×8 Snake game supplies an 11-value state: danger straight/right/left,
  current direction, and food direction.
- The agent chooses straight, right, or left. A network with one 16-unit ReLU
  hidden layer estimates the value of each action.
- Online Q-learning updates the network after every move. Exploration falls from
  100% to 5% during training.
- Evaluation uses 100 previously unseen game seeds and compares the trained
  agent with random actions under the same 200-move limit.

On this machine, 3,000 training games produced these results:

| Training seed | Trained agent | Random actions |
| --- | ---: | ---: |
| 0 | 15.28 apples/game | 0.17 apples/game |
| 42 | 16.23 apples/game | 0.20 apples/game |

These are observed results, not a guarantee for every seed or a general Snake
benchmark. Rewards include a small distance-to-food signal. The network uses
online updates without experience replay or a separate target network. Evaluation
stops after 200 moves, even when the snake is still alive. The saved JSON model is
for this experiment and is not compatible with the old TensorFlow model.
