import os
import json
import torch


y = torch.tensor ([[[1],[2],[3]],[[1],[2],[3]]])
s = y.shape
lis = y.tolist()

newy = torch.tensor(lis)
print (y)
print (y.shape)
print (newy)
print (newy.shape)


flat = y.flatten()
print (flat)


with open("testesto.json", "w") as fp:
    json.dump(lis, fp)
    print ("dumped")


# with open("student.json", "w") as fp:
#     for x in os.listdir("images"):
#         json.dump(x, fp)


# dicto = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
# for x in dicto:
#     print (x)