from env.Model import model
import numpy as np
import pygame

grid = np.load('saved_env/env.npy')

model = model("Maze",grid)
#model = model("Maze")

pygame.init()
stop = False

while(stop!=True):
    stop = model.show()
    model.performe_action(action="down")
    model.performe_action(action="right")
    # model.performe_action(action="")
    # model.performe_action(action="down")

