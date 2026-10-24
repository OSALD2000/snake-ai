import random


DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))
TURNS = (0, 1, -1)  # straight, right, left


class SnakeGame:
    def __init__(self, size=8, seed=0):
        if size < 4:
            raise ValueError("size must be at least 4")
        self.size = size
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        center = self.size // 2
        self.body = [(center, center), (center - 1, center)]
        self.direction = 1
        self.score = 0
        self.done = False
        self.food = self._new_food()
        return self.state()

    def _new_food(self):
        free = [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in self.body]
        return self.random.choice(free) if free else None

    def _next_head(self, turn):
        direction = (self.direction + turn) % 4
        dx, dy = DIRECTIONS[direction]
        x, y = self.body[0]
        return (x + dx, y + dy)

    def _blocked(self, head):
        x, y = head
        growing = head == self.food
        return (x < 0 or x >= self.size or y < 0 or y >= self.size
                or head in (self.body if growing else self.body[:-1]))

    def state(self):
        head_x, head_y = self.body[0]
        food_x, food_y = self.food if self.food is not None else self.body[0]
        return tuple(float(value) for value in (
            *(self._blocked(self._next_head(turn)) for turn in TURNS),
            *(self.direction == direction for direction in range(4)),
            food_x < head_x, food_x > head_x,
            food_y < head_y, food_y > head_y,
        ))

    def step(self, action):
        if self.done:
            raise ValueError("reset before the next step")
        if action not in range(3):
            raise ValueError("action must be 0, 1 or 2")

        head = self._next_head(TURNS[action])
        if self._blocked(head):
            self.done = True
            return self.state(), -10.0, True

        old_distance = abs(self.body[0][0] - self.food[0]) + abs(self.body[0][1] - self.food[1])
        self.direction = (self.direction + TURNS[action]) % 4
        ate = head == self.food
        self.body.insert(0, head)
        if ate:
            self.score += 1
            self.food = self._new_food()
            self.done = self.food is None
            reward = 10.0
        else:
            self.body.pop()
            new_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            reward = 0.1 if new_distance < old_distance else -0.1
        return self.state(), reward, self.done
