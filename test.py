import crypten
import torch
crypten.init()

x = torch.tensor([[1,2,3],[4,5,6]])
enc_x = crypten.cryptensor(x)
enc_x._tensor.data = enc_x._tensor.data.contiguous()
print(0)
