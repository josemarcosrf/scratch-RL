from typing import Optional

from pydotdict import DotDict as dotdict


class Random1DWalk:

    """This is the implmentatin of the environment described in:
    'CHAPTER 6. TEMPORAL-DIFFERENCE LEARNING - Example 6.2 Random Walk' (page 100)
    of Sutton and Barto's book: 'Reinforcement Learning An Introduction'

    Returns:
        _type_: _description_
    """

    ACTIONS = {-1: "left", 1: "right"}

    def __init__(self, size: int = 100):
        # if size % 2 == 0:
        #     size += 1
        self.size = size
        self.pos = size // 2  # starting position
        self.grid = ["o"] * size

    @property
    def state_space(self):
        return dotdict({"n": self.size})

    @property
    def action_space(self):
        return dotdict({"n": self.size})

    def _within_boundaries(self):
        return self.pos >= 0 and self.pos < len(self.grid)

    def reset(self, pos: Optional[int] = None):
        self.pos = pos if pos is not None else len(self.grid) // 2
        return self.pos, {}

    def step(self, action: int):
        assert action in self.ACTIONS, f"Action must be in: {list(self.ACTIONS.keys())}"
        assert self._within_boundaries(), "agent outside the world. reset environment?"

        # self.pos = max(0, min(self.pos, len(self.grid)))
        self.pos += action

        reward, terminated = 0, False
        if self.pos == -1:
            reward = -1
            terminated = True
        elif self.pos == len(self.grid):
            reward = 1
            terminated = True

        return self.pos, reward, terminated, False, {}

    def render(self):
        if self._within_boundaries():
            self.grid = ["o"] * len(self.grid)
            self.grid[self.pos] = "*"
        elif self.pos < 0:
            self.grid = ["<"] * len(self.grid)
        else:
            self.grid = [">"] * len(self.grid)

        print(self.grid)


if __name__ == "__main__":
    env = Random1DWalk(size=10)
    env.render()
