import numpy as np
logits = np.random.randn(5,3)
targets = np.random.randint(0,high=2,size=5)
axis=1

num_features = np.size(logits,axis)
num_batches = np.size(targets)
one_hot = np.zeros((num_batches,num_features))
rows = np.arange(num_batches)
one_hot[rows,targets] = 1
print(targets)
print(one_hot)

log_logit_2 = logits - np.log(np.sum(np.exp(logits),axis=axis,keepdims=True))
#print(logits)
logits = logits - np.amax(logits,axis=axis,keepdims=True)
#print(logits)
logits = np.exp(logits)
temp = np.sum(logits,axis=axis,keepdims=True)
#print(temp)
logits = logits/temp
log_logits = np.log(logits)
H = np.zeros(targets.shape)
for i in range (0,targets.size):
    H[i] = -1*log_logits[i,targets[i]]
    
c = -1*log_logits*one_hot
