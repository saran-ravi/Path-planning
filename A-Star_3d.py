

from queue import PriorityQueue
import numpy as np
from enum import Enum


class Action(Enum):
    """
    An action is represented by a 3 element tuple.
    
    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """
    LEFT = (0, 0, -1, 1)
    RIGHT = (0, 0, 1, 1)
    UP = (0, -1, 0, 1)
    DOWN = (0, 1, 0, 1)
    IN = (1, 0, 0, 1)
    OUT = (-1, 0, 0, 1)
    
    def __str__(self):
        if self == self.LEFT:
            return '<'
        elif self == self.RIGHT:
            return '>'
        elif self == self.UP:
            return '^'
        elif self == self.DOWN:
            return 'v'
        elif self == self.IN:
            return '*'
        elif self == self.OUT:
            return '.'
    
    @property
    def cost(self):
        return self.value[3]
    
    @property
    def delta(self):
        return (self.value[0], self.value[1], self.value[2])
            
    
def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid = [Action.UP, Action.LEFT, Action.RIGHT, Action.DOWN, Action.IN, Action.OUT]
    o, n, m = grid.shape[0] - 1, grid.shape[1] - 1, grid.shape[2] - 1
    z, x, y = current_node
    # print("z, x, y = ", z, x, y)
    # print("n, m, o = ", n, m, o)
    # check if the node is off the grid or
    # it's an obstacle
    
    # if x - 1 < 0 or grid[x-1, y, z] == 1:
    #     valid.remove(Action.UP)
    # if x + 1 > n or grid[x+1, y, z] == 1:
    #     valid.remove(Action.DOWN)
    # if y - 1 < 0 or grid[x, y-1, z] == 1:
    #     valid.remove(Action.LEFT)
    # if y + 1 > m or grid[x, y+1, z] == 1:
    #     valid.remove(Action.RIGHT)
    # if z - 1 < 0 or grid[x, y, z-1] == 1:
    #     valid.remove(Action.IN)
    # if z + 1 > o or grid[x, y, z+1] == 1:
    #     valid.remove(Action.OUT)

    # print("In valid actions = ",current_node)

    if x - 1 < 0 or grid[z, x-1, y] == 1:
        valid.remove(Action.UP)
    if x + 1 > n or grid[z, x+1, y] == 1:
        valid.remove(Action.DOWN)
    if y - 1 < 0 or grid[z, x, y-1] == 1:
        valid.remove(Action.LEFT)
    if y + 1 > m or grid[z, x, y+1] == 1:
        valid.remove(Action.RIGHT)
    if z - 1 < 0 or grid[z-1, x, y] == 1:
        valid.remove(Action.OUT)
    if z + 1 > o or grid[z+1, x, y] == 1:
        valid.remove(Action.IN)
    # print("valid = ", valid)

    
    return valid

def visualize_path(grid, path, start):
    sgrid = np.zeros(np.shape(grid), dtype=np.str)
    sgrid[:] = ' '
    sgrid[grid[:] == 1] = 'O'
    
    pos = start
    
    for a in path:
        # a has enumerated variable
        da = a.value
        sgrid[pos[0], pos[1], pos[2]] = str(a)
        pos = (pos[0] + da[0], pos[1] + da[1], pos[2] + da[2])
    sgrid[pos[0], pos[1], pos[2]] = 'G'
    sgrid[start[0], start[1], start[2] ] = 'S'  
    return sgrid


# ## Heuristics
# The heuristic function determines the $h()$ value for each cell based on the goal cell and the method chosen to determine it. The heuristic value can be the Euclidean distance between these cells $h= \left((x_i-x_{goal})^2+(y_i-y_{goal})^2\right)^{1/2}$ or the "Manhattan distance", which is the minimum number of moves required to reach the goal from the assigned cell $h = ||x_i-x_{goal}|| + ||y_i-y_{goal}||$. For this exercise you could use either, or something else which is *admissible* and *consistent*.
# 
# The input variables include
# * **```position```** the coordinates of the cell for which you would like to determine the heuristic value.
# * **```goal_position```** the coordinates of the goal cell

# In[4]:


# TODO: implement a heuristic function. This may be one of the
# functions described above or feel free to think of something
# else.
def heuristic(position, goal_position):
    h = 0
    h = (position[0] - goal_position[0])**2 + abs(position[1] - goal_position[1])**2 + abs(position[2] - goal_position[2])**2
    h = h*20
    #h = abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1])
    return h


# ## A* search
# 
# A* search is an extension of the cost search you implemented. A heuristic function is used in addition to the cost penalty. Thus if the setup is:
# 
# * $c$ is the current cost
# * $g$ is the cost function
# * $h$ is the heuristic function
# 
# Then the new cost is $c_{new} = c + g() + h()$.
# 
# The difference between $g$ and $h$ is that $g$ models the cost of performing actions, irrespective of the environment, while $h$ models the cost based on the environment, i.e., the distance to the goal.

# You know what comes next, turn the `TODOs` into `DONEs` :)

# In[5]:


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():

        # gives the item which has min cost = (branch + heuristics)
        item = queue.get()
        # print("item = ", item)

        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            ## takes only branch cost and not heuristics              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1], current_node[2] + da[2])
#                 # TODO: calculate branch cost (action.cost + g)
#                 g = current_cost + action.cost
#                 # TODO: calculate queue cost (action.cost + g + h)
#                 f = g + h(next_node, goal)
#                 branch_cost = g
#                 queue_cost = f
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))    
                    
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            # path stores only the action
            path.append(branch[n][2])
            n = branch[n][1]
        path.append(branch[n][2])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        
    return path[::-1], path_cost
    #h = abs(position[0] - global_position[0]) + abs(position[1] - position[1])


# In[6]:


start = (0,4,0)
# goal = (1,4,4)
goal = (2,1,1)

# grid1 = np.array([
#     [0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 0, 0],
# ])

# # print(np.shape(grid1))
# grid = np.array(
#     [[[0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 0, 0]], 
#     [[0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 1, 0],
#     [0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 0, 0],]]
#)

grid = np.array(
    [[[0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0]], 
    [[1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],],
    [[1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0],]])
print("start = ", start)
print("goal = ", goal)
# print((grid[0, 1 ,2]))
# In[7]:
print(goal)
input("stop")

path, cost = a_star(grid, heuristic, start, goal)
# print(path, cost)

for i in grid:
    for j in i:

        print(j)
        

# In[8]:


# S -> start, G -> goal, O -> obstacle
print(visualize_path(grid, path, start))


# [Solution](/notebooks/A-Star-Solution.ipynb)

# In[ ]:




