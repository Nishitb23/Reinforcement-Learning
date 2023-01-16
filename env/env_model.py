import pygame
import numpy as np


class model():

    def __init__(self,*args) -> None:
        
        # self.width = 1540
        # self.height = 780
        # self.block = 10
        self.width = 120
        self.height = 100
        self.block = 20
        self.agent_pos = (0,0)
        self.screen = pygame.display.set_mode((self.width,self.height))
        self.screen.fill((255,255,255))
        self.action_list = [0,1,2,3]
        self.state_length = (self.width//self.block)*(self.height//self.block)
        pygame.init()
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
            self.grid[self.grid.shape[0]-1][self.grid.shape[1]-1] = 5/7

            #placing agent on grid
            self.grid[0][0] = 6/7

            #copying intial enviroment for rollouts
            self.initial_grid = np.copy(self.grid)
            
            #save the enviroment
            np.save('saved_env/env',self.grid)
        else:
            self.grid = args[1]
            self.initial_grid = args[1]
            
                
        
    def place_walls(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        wall_states = int(0.1*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 2/7


    def place_fruits(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        wall_states = int(0.008*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=wall_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=wall_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 4/7


    def place_restarts(self)->None:

        total_states = self.grid.shape[0]*self.grid.shape[1]
        restart_states = int(0.01*total_states)
        rows = 1+np.random.randint(self.grid.shape[0]-2,size=restart_states)
        cols = 1+np.random.randint(self.grid.shape[1]-2,size=restart_states)   
        pos = zip(rows,cols)

        for x,y in pos:
            self.grid[x][y] = 3/7


    def draw_grid(self)->None:
        
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                if self.grid[x][y] == 1: #empty
                    pygame.draw.rect(self.screen, (0,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y] == 2/7: #wall
                    pygame.draw.rect(self.screen, (255,255,255), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y] == 3/7: # restart
                    pygame.draw.rect(self.screen, (255,0,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y] == 4/7: # fruit
                    pygame.draw.rect(self.screen, (255,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y] == 5/7: #goal
                    pygame.draw.rect(self.screen, (0,255,0), (x*self.block,y*self.block,(x+1)*self.block, 
                (y+1)*self.block))
                elif self.grid[x][y] == 6/7: #agent
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
        


    def reset(self):
        self.grid = np.copy(self.initial_grid)
        self.agent_pos = (0,0)
        return self.grid.flatten()


    def perform_action(self,action):

        reward = 0
        done = False
        new_x = self.agent_pos[0]
        new_y = self.agent_pos[1]

        if action == 0: #right monement
            
            new_x = new_x + 1
            if new_x == self.width//self.block:
                return self.grid.flatten(), reward, done
                
        elif action == 1: #left movement
            
            new_x = new_x - 1
            if new_x == -1:
                return self.grid.flatten(), reward, done

        elif action == 2: #up movement
            
            new_y = new_y - 1
            if new_y == -1:
                return self.grid.flatten(), reward, done

        elif action  == 3: #down movement

            new_y = new_y + 1
            if new_y == self.height//self.block:
                return self.grid.flatten(), reward, done

        else:
            print("Invalid action")

        if(self.grid[new_x][new_y] == 2/7):
            return self.grid.flatten(), reward, done
        elif(self.grid[new_x][new_y] == 3/7):
            reward = -40
            self.reset()
            return self.grid.flatten(), reward, done
        elif(self.grid[new_x][new_y] == 4/7):
            reward = 5
        elif(self.grid[new_x][new_y] == 5/7):
            reward = 100
            done = True

        self.grid[self.agent_pos[0]][self.agent_pos[1]] = 1
        self.grid[new_x][new_y] = 6/7
        self.agent_pos = (new_x,new_y)

        return self.grid.flatten(), reward, done