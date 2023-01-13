from env.env_model import model
import numpy as np
import sys

path = str(sys.argv[0])
print(path)
# if path == None:
#     model = model("Maze")
# else:
#     grid = np.load(path,allow_pickle=True)
#     model = model("Maze",grid)

grid = np.load('saved_env/env.npy')
#model = model("Maze",grid)
model = model("Maze")

stop = False

while(stop!=True):
    stop = model.show()
    # model.performe_action(action="down")
    # model.performe_action(action="right")
    # model.performe_action(action="down")

