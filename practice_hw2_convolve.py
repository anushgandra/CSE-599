import numpy as np

batches = 10;
in_ch = 6;
h = 9;
w = 10;

kernel_size = 7;
padding = (kernel_size - 1)//2

inputs = np.ones((batches,in_ch,h,w),dtype=np.float32)

out = np.pad(inputs,((0,0),(0,0),(padding,padding),(padding,padding)),'constant');

##print(out[0][0])
##
##print(inputs[0][0])


mat = out[0][0]

print(mat)
print(out.shape)
print(inputs.shape)
