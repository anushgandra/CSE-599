import numpy as np

slope = 0.01

inputs = np.array(((1,2,3),(4,-3,-3),(-1,4,-1),(-4,7,1)),dtype=np.float32)
print(inputs)
inp2 = inputs
print(inp2)
inputs[inputs<=0] = slope*inputs[inputs<=0]
#print(inputs)
#inp2[inp2<=0] = slope
#inp2[inp2>0] = 1
