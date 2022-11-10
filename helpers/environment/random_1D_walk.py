class Random1DWalk:

    ACTIONS = {
        -1: "left",
        1: "right"
    }

    def __init__(self, size: int = 100):
        if size % 2 == 0:
            size += 1
        self.pos = size // 2  # starting position
        self.grid = ["o"] * size

    def reset(self):
        self.pos = len(self.grid) // 2
        return self.pos, {}

    def step(self, action: int):
        assert action in self.ACTIONS, f"Action must be in: {list(self.ACTIONS.keys())}"

        self.pos += action
        self.pos = max(min(self.pos, len(self.grid) + 1), 0)

        if self.pos == 0:
            reward = -1
            terminated = True
        elif self.pos == len(self.grid):
            reward = 1
            terminated = True

        return self.pos, reward, terminated, False, {}

    def render(self):
        self.grid = ["o"] * len(self.grid)
        self.grid[self.pos] = "*"
        print(self.grid)


if __name__ == "__main__":
    env =Random1DWalk(size=10)
    env.render()
