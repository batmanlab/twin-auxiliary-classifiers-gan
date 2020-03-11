import numpy as np
import torch
data = np.load('omniglot_data.npy')

print(data.shape)

imgs = []
labels = []

for cls in range(data.shape[0]):

    for d in range(data.shape[1]):
        imgs.append(data[cls,d])
        labels.append(cls)

imgs = torch.from_numpy(np.asarray(imgs))
labels = torch.from_numpy(np.asarray(labels))

torch.save([imgs,labels], 'omniglot.pt')

