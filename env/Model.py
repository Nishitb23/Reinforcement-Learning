import pygame
import numpy as np


class model():

    def __init__(self,*args) -> None:
        
        self.width = 1000
        self.height = 780
        self.block = 20
        self.agent_pos = (0,0)
        self.screen = pygame.display.set_mode((self.width,self.height))
        self.screen.fill((255,255,255))
        pygame.display.set_caption(args[0])

        if len(args)==1:
            self.grid = np.ones((self.width//self.block,self.height//self.block))
        
            #change grid by loading wall
            self.place_walls()

            #change grid by placing restart
            self.place_restarts()

            #change grid by placing fruit
            self.place_fruits()

            #placing goal on grid
            self.grid[self.grid.shape[0]-1][self.grid.shape[1]-1] = 5

            #placing agent on grid
            self.grid[0][0] = 6
            
            #save the enviroment
            np.save('saved_env/env',self.grid)
        else:
            self.grid = args[1]
                
        
    def place_walls(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        wall_states = int(0.1*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 2


    def place_fruits(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        wall_states = int(0.008*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 4


    def place_restarts(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        restart_states = int(0.01*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=restart_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=restart_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 3


    def draw_grid(self)->None:
        
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y]==1: #empty
                    pygame.draw.rect(self.screen, (0,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==2: #wall
                    pygame.draw.rect(self.screen, (255,255,255), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==3: # restart
                    pygame.draw.rect(self.screen, (255,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==4: # fruit
                    pygame.draw.rect(self.screen, (255,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==5: #goal
                    pygame.draw.rect(self.screen, (0,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y]==6: #agent
                    pygame.draw.rect(self.screen, (0,0,255), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
        

    def show(self) -> None:
 
        clock = pygame.time.Clock()
        stop = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
            if event.type == pygame.MOUSEBUTTONUP:
                stop = True
        self.draw_grid()
        pygame.display.update()
        clock.tick(7)
        return stop

    def get_reward(self)->int:
        
        next_pos = self.agent_pos
        if(self.grid[next_pos[0]][next_pos[1]] == 3):
            return -100
        elif(self.grid[next_pos[0]][next_pos[1]] == 4):
            return 5
        elif(self.grid[next_pos[0]][next_pos[1]] == 5):
            return 100
        else:
            return 0

    def performe_action(self,action="down"):

        if action.lower()=="right":
            if (self.agent_pos[0]+1)==self.width//self.block:
                return self.agent_pos
            else:
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 1
                self.agent_pos = (self.agent_pos[0]+1,self.agent_pos[1])
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 6
                
        elif action.lower()=="left":
            if self.agent_pos[0]==0:
                return self.agent_pos
            else:
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 1
                self.agent_pos = (self.agent_pos[0]-1,self.agent_pos[1])
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 6
                return self.agent_pos
        elif action.lower()=="up":
            if self.agent_pos[1]==0:
                return self.agent_pos
            else:
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 1
                self.agent_pos = (self.agent_pos[0],self.agent_pos[1]-1)
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 6
                return self.agent_pos
        elif action.lower()=="down":
            if self.agent_pos[1]+1==self.height//self.block:
                return self.agent_pos
            else:
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 1
                self.agent_pos = (self.agent_pos[0],self.agent_pos[1]+1)
                self.grid[self.agent_pos[0]][self.agent_pos[1]] = 6
        else:
            print("Invalid action")
            return self.agent_pos