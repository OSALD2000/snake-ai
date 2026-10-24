import argparse
import random

from .agent import QNetwork
from .game import SnakeGame


def run_game(agent, seed, epsilon=0.0, learn=False, max_steps=200):
    game = SnakeGame(seed=seed)
    rng = random.Random(seed + 1_000_000)
    state = game.state()
    for step in range(max_steps):
        action = rng.randrange(3) if rng.random() < epsilon else agent.act(state)
        next_state, reward, done = game.step(action)
        if learn:
            agent.learn(state, action, reward, next_state, done or step + 1 == max_steps)
        state = next_state
        if done:
            break
    return game.score


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a small Snake Q-network")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="model.json")
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_games < 1:
        parser.error("episodes and eval-games must be positive")

    agent = QNetwork(seed=args.seed)
    scores = []
    for episode in range(args.episodes):
        epsilon = max(0.05, 1.0 - episode / (args.episodes * 0.8))
        scores.append(run_game(agent, args.seed + episode, epsilon=epsilon, learn=True))
    agent.save(args.model)

    seeds = range(args.seed + 100_000, args.seed + 100_000 + args.eval_games)
    learned = [run_game(agent, seed) for seed in seeds]
    random_scores = [run_game(agent, seed, epsilon=1.0) for seed in seeds]
    print(f"Last 100 training games: {sum(scores[-100:]) / len(scores[-100:]):.2f} apples/game")
    print(f"Trained agent: {sum(learned) / len(learned):.2f} apples/game")
    print(f"Random actions: {sum(random_scores) / len(random_scores):.2f} apples/game")
    print(f"Model saved to {args.model}")


if __name__ == "__main__":
    main()
