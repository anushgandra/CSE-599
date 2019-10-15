import numpy as np

slopes = np.array((1.1,2.1,3.1),dtype=np.float32)
inputs = np.array(((1, -2, 5),(2,3,-1),(3,-2,5),(-1,9,0)),dtype=np.float32)

#print(inputs)

out1 = np.copy(inputs)
out2 = np.copy(inputs)

out1[out1>0] = 0

out1 = out1*slopes

out2[out2<=0] = 0 

##print(out1)
##print(out2)

out = out1+out2

#print(out)

out3 = np.copy(inputs)
out3[out3<=0] = 0

##print(inputs)
##print(out3)

out4 = np.copy(inputs)
temp1 = np.argwhere(out4>0)
##print(out4)
##print(inputs)
out4[out4<=0] = 1
out4 = out4*slopes

out4[temp1[:,0],temp1[:,1]] = 0

grad = np.copy(inputs)
grad[grad>0] = 0

print(inputs)
print(grad)

out_2 = out3+out4
##print(inputs)
##print(slopes)
##print(out)
##print(out_2)


