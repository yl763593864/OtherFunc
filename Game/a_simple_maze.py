import pygame
from enum import Enum
import numpy as np


class MoveType(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Maze:
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 600
        self.bg_color = (58, 85, 82)
        self.green_img = pygame.image.load('Img/green.png')
        self.white_img = pygame.image.load('Img/white.png')
        self.reset_pos()

    def reset_pos(self):
        self.green_pos = [np.random.randint(0, 8)*100, np.random.randint(0, 6)*100]
        self.white_pos = [np.random.randint(0, 8)*100, np.random.randint(0, 6)*100]
        while True:
            if self.white_pos == self.green_pos:
                self.white_pos = [np.random.randint(0, 7) * 100, np.random.randint(0, 7) * 100]
            else:
                break


    def run_game(self):
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('a simple maze')
        clock = pygame.time.Clock()

        crashed = False
        while not crashed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    crashed = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.move(MoveType.Left)
                    elif event.key == pygame.K_s:
                        self.move(MoveType.Down)
                    elif event.key == pygame.K_d:
                        self.move(MoveType.Right)
                    elif event.key == pygame.K_w:
                        self.move(MoveType.Up)
            self.screen.fill(self.bg_color)
            self.display()
            pygame.display.flip()
            clock.tick(60)

    def display(self):
        self.screen.blit(self.white_img, (self.white_pos[0], self.white_pos[1]))
        self.screen.blit(self.green_img, (self.green_pos[0], self.green_pos[1]))

    def move(self, move_type: MoveType):
        if move_type == MoveType.Down:
            if self.white_pos[1] + 200 <= self.screen_height:
                self.white_pos[1] += 100
        elif move_type == MoveType.Up:
            if self.white_pos[1] - 100 >= 0:
                self.white_pos[1] -= 100
        elif move_type == MoveType.Left:
            if self.white_pos[0] - 100 >= 0:
                self.white_pos[0] -= 100
        elif move_type == MoveType.Right:
            if self.white_pos[0] + 200 <= self.screen_width:
                self.white_pos[0] += 100

        if self.green_pos == self.white_pos:
            self.reset_pos()


if __name__ == '__main__':
        maze = Maze()
        maze.run_game()

