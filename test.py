import sys

import crypten
import torch
crypten.init()

#x = torch.rand((10,104,512),dtype = torch.float64)
#y = torch.rand((10,104,1),dtype = torch.float64)
x = torch.randint(1000000000000000000,9000000000000000000,(10,104,512))
y = torch.randint(1000000000000000000,9000000000000000000,(512,2048))
#intx = 22200000000000000000
#print(sys.getsizeof(x[0][0][0]))
#y = 2

import timeit
t_start = timeit.default_timer()
result = torch.matmul(x,y)
#result = torch.matmul(x,y)
t_end = timeit.default_timer()
print("%f" % (t_end-t_start))
