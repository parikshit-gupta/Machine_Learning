from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import deque

class Rewards(Enum):
    food = 1
    step = -0.01
    death = -1
    
class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

"""
- This snake does not move with every tick, rather it moves when the agent takes an action.
- This design of the environment allows for more control over the agent's actions and makes it easier to 
    train reinforcement learning models.
- The agent can take one of four actions: right, up, left, or down.
"""
class ClassicSnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, size=10):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        #Observation space is a square grid of size `size x size`
        # with values 0 (empty), 1 (snake), and 2 (food)
        self.observation_space = spaces.Box(low=0, high=2, shape=(size, size), dtype=np.int8)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        
        # describing internal states of the environment
        self.snake=deque()
        """
        leftmost element of the deque is the head of the snake,
        rightmost element is the tail of the snake.
        """
        self.food=None
        self.grid=None
        self.direction=None
        self.score=0
        self.step_count=0
        self.step_limit=size*2.5
        
        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def _get_obs(self):
        grid_copy=np.copy(self.grid)
        return grid_copy
    
    def _get_info(self):
        return {"score": self.score}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.snake.clear()
        self.snake.append((self.size // 2, self.size // 3))
        
        self.food = (self.size // 2, self.size-(self.size // 3))
        
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.grid[self.snake[0]] = 1
        self.grid[self.food] = 2
        
        self.direction = self._action_to_direction[0]  # Start moving right
        self.score = 0
        self.step_count = 0
        
        observation= self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        
        self.step_count += 1
        truncate=False
        if(self.step_count >= self.step_limit):
            truncate=True
        
        # 1) check is action is valid cant be the opposite of the current direction
        direction=self._action_to_direction[action]
        if self.score!=0 and self.direction[0]+direction[0] == 0 and self.direction[1]+direction[1] == 0:
            direction=self.direction  # Ignore the action if it's the opposite direction
        self.direction=direction
        
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        # 2) Snake has hit the wall or itself
        if new_head[0] < 0 or new_head[0] >= self.size or \
           new_head[1] < 0 or new_head[1] >= self.size or \
           self.grid[new_head] == 1:
            return self._get_obs(), Rewards.death.value, True, truncate, self._get_info()
        
        # 3) Snake has eaten the food
        if self.grid[new_head] == 2:
            self.step_count=0
            self.snake.appendleft(new_head)  # Add new head to the front of the snake
            self.grid[new_head] = 1
            
            #spawn new food till it does not spawn on the snake
            new_food=self.food
            while self.grid[new_food]==1:
                new_food = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            self.food=new_food
            self.grid[self.food] = 2
            
            self.score+=1

            return self._get_obs(), Rewards.food.value, False, truncate, self._get_info()
                        
        self.snake.appendleft(new_head) # Add new head to the front of the snake
        tail=self.snake.pop()  # Remove the tail
        self.grid[new_head] = 1
        self.grid[tail] = 0
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self._get_obs(), Rewards.step.value, False, truncate, self._get_info()
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        boundry_gap=6
        gridlines_gap=2
        WHITE= (255, 255, 255)
        BLACK= (0, 0, 0)
        RED= (255, 0, 0)
        BLUE= (0, 0, 255)
        GREEN= (0, 255, 0)
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)
        pix_square_size = (
            (self.window_size-2*boundry_gap) / self.size   # 12 pixels for boundry, 22 pixels for gridlines are left out
        )  # The size of a single grid square in pixels
        
        #mini grid square where the snake and food are drawn
        mini_grid_size=pix_square_size - 2*gridlines_gap  # Adjust for gridlines

        #draw boundary recatangle
        pygame.draw.rect(
            canvas,
            BLACK,
            pygame.Rect(0, 0, self.window_size, self.window_size),
            width=boundry_gap,
        )
        
        # First we draw the food
        pygame.draw.rect(
            canvas,
            RED,
            pygame.Rect(
                pix_square_size * self.food[1] +gridlines_gap+ boundry_gap,
                pix_square_size * self.food[0] +gridlines_gap+ boundry_gap,
                mini_grid_size, mini_grid_size,
            ),
        )
        # Now we draw the agent
        for segment in self.snake:
            pygame.draw.rect(
                canvas,
                BLACK,
                pygame.Rect(
                    pix_square_size * segment[1] + gridlines_gap + boundry_gap,
                    pix_square_size * segment[0] + gridlines_gap + boundry_gap,
                    mini_grid_size,mini_grid_size,
                ),
            )
        
        # for snake's head
        pygame.draw.circle(
                canvas,
                WHITE,
                (pix_square_size * (self.snake[0][1]+.5) + boundry_gap, pix_square_size * (self.snake[0][0]+.5) + boundry_gap),
                100/self.size,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()