import torch
import torch.nn as nn
import math
"""
m=nn.Softmax(dim=1)
input=torch.randn(2,2)
output=m(input)
fenmu=math.exp(input[1][0])+math.exp(input[1][1])
print(fenmu)
print(math.exp(input[1][0])/fenmu)
print(input)
print(output)
"""

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
print(output)