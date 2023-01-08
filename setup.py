from env.Model import model
import numpy as np

grid = np.load('env.npy')

model = model(grid,title="Maze")

model.start()
