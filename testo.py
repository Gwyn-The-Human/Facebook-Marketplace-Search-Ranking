import os
import json
import torch


with open("images/0a1d0925-d2aa-4e89-b9d3-ef56b834cfd9.jpg","rb") as bipic:
    data = bipic.read()
    print (type(data))


# y = torch.tensor ([[[1],[2],[3]],[[1],[2],[3]]])
# s = y.shape
# lis = y.tolist()

# newy = torch.tensor(lis)
# print (y)
# print (y.shape)
# print (newy)
# print (newy.shape)


# flat = y.flatten()
# print (flat)


# with open("testesto.json", "w") as fp:
#     json.dump(lis, fp)
#     print ("dumped")


# with open("student.json", "w") as fp:
#     for x in os.listdir("images"):
#         json.dump(x, fp)


# dicto = {'color': 'blue', 'fruit': 'apple', 'pet': 'dog'}
# for x in dicto:
#     print (x)