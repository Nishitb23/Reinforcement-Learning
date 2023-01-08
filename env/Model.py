import pygame
import numpy as np


class model():

    def __init__(self,title="sample game") -> None:
        
        self.width = 1000
        self.height = 780
        self.block = 20
        self.grid = np.ones((self.width//self.block,self.height//self.block))
        self.screen = pygame.display.set_mode((self.width,self.height))
        self.screen.fill((255,255,255))
        self.goal = (self.grid.shape[0],self.grid.shape[1])
        
        pygame.display.set_caption(title)

        #change grid by loading wall
        self.grid = self.get_wall(self.grid)
        
    def get_wall(self,grid):

        total_states = grid.shape[0]*grid.shape[1]
        wall_states = int(0.1*total_states)
        rows = 1+np.random.randint(grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            grid[x][y] = 2

        return grid

    def draw_grid(self)->None:
        
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y]==1:
                    pygame.draw.rect(self.screen, (0,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                if self.grid[x][y]==2:
                    pygame.draw.rect(self.screen, (255,255,255), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
        

    def start(self) -> None:
        pygame.init()
        clock = pygame.time.Clock()
        stop = False
        while(stop != True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
        	        stop = True
                if event.type == pygame.MOUSEBUTTONUP:
        	        stop = True
            self.draw_grid()
            pygame.display.update()
            clock = pygame.time.Clock()
