import pygame
from enum import Enum
import numpy as np
import Game.game_training as NN

class MoveType(Enum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3


class Maze:
    def __init__(self, nn):
        self.screen_width = 600
        self.screen_height = 600
        self.bg_color = (58, 85, 82)
        self.green_img = pygame.image.load('Img/green.png')
        self.white_img = pygame.image.load('Img/white.png')
        self.reset_pos()
        self.nn = nn

    def reset_pos(self):
        self.green_pos = [np.random.randint(0, 6)*100, np.random.randint(0, 6)*100]
        self.white_pos = [np.random.randint(0, 6)*100, np.random.randint(0, 6)*100]
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
                    elif event.key == pygame.K_SPACE:
                        self.nn_control_move()
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

    def nn_control_move(self):
        data, tart = self.generate_data()
        num = self.nn.test(data, tart)
        move_type = MoveType(num)
        self.move(move_type)

    def generate_data(self):
        game_array = np.zeros((8, 8))
        for i in np.arange(8):
            for j in np.arange(8):
                if i == 0 or j == 0 or j == 7 or j == 7:
                    game_array[i][j] = 0
                else:
                    game_array[i][j] = 0.2

        zero_pos = [self.white_pos[0]//100, self.white_pos[1]//100]
        target_pos = [self.green_pos[0]//100, self.green_pos[1]//100]
        game_array[zero_pos[0] + 1, zero_pos[1] + 1] = 0.8
        game_array[target_pos[0] + 1, target_pos[1] + 1] = 1

        target_array = np.zeros((4, 1))
        pos = [target_pos[0] - zero_pos[0], target_pos[1] - zero_pos[1]]
        print('pos', pos)
        if pos[1] >= 0:
            target_array[0] = 0
            target_array[1] = abs(pos[1])
        else:
            target_array[0] = abs(pos[1])
            target_array[1] = 0

        if pos[0] >= 0:
            target_array[2] = 0
            target_array[3] = abs(pos[0])
        else:
            target_array[2] = abs(pos[0])
            target_array[3] = 0


        max_num = max(target_array)
        print('max num', max_num, target_array)
        for i in np.arange(4):
            if target_array[i][0] <= 0:
                target_array[i][0] = 0
            else:
                if target_array[i] == max_num:
                    target_array[i] = 1
                else:
                    target_array[i] /= 6

        return game_array, target_array




if __name__ == '__main__':
        nn =NN.NeuralNetwork()
        nn.read_data()
        maze = Maze(nn)
        maze.run_game()

