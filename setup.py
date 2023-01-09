from env.Model import model
import numpy as np

grid = np.load('saved_env/env.npy')

model = model("Maze",grid)
#model = model("Maze")

model.start()
