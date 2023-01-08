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

        #change grid by placing restart
        self.grid = self.get_restart(self.grid)

        #change grid by placing fruit
        self.grid = self.get_fruit(self.grid)

        #placing goal on grid
        self.grid[self.grid.shape[0]-1][self.grid.shape[1]-1] = 5

        #placing agent on grid
        self.grid[0][0] = 6
        
    def get_wall(self,grid):

        total_states = grid.shape[0]*grid.shape[1]
        wall_states = int(0.1*total_states)
        rows = 1+np.random.randint(grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            grid[x][y] = 2

        return grid

    def get_fruit(self,grid):

        total_states = grid.shape[0]*grid.shape[1]
        wall_states = int(0.008*total_states)
        rows = 1+np.random.randint(grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            grid[x][y] = 4

        return grid

    def get_restart(self,grid):

        total_states = grid.shape[0]*grid.shape[1]
        restart_states = int(0.01*total_states)
        rows = 1+np.random.randint(grid.shape[0]-2,size=restart_states)
        cols = 1+np.random.randint(grid.shape[1]-2,size=restart_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            grid[x][y] = 3

        return grid

    def draw_grid(self)->None:
        
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y]==1:
                    pygame.draw.rect(self.screen, (0,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==2:
                    pygame.draw.rect(self.screen, (255,255,255), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==3:
                    pygame.draw.rect(self.screen, (255,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==4:
                    pygame.draw.rect(self.screen, (255,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==5:
                    pygame.draw.rect(self.screen, (0,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==6:
                    pygame.draw.rect(self.screen, (0,0,255), (x*self.block,y*self.block,(x+1)*self.block, 
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
