import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torchvision
from tqdm import tqdm
import numpy as np
#from datasets import load_dataset
from PIL import Image
from models.ResNet import ResNet18
import pickle
import os
import sys

"""
Code of a single agent training
"""

from config import TRAIN_SPLIT, DATASET, NUM_NODES, CHUNKING, BATCH_SIZE, TEST_SPLIT, NUM_EPOCHS, SEED
DEVICE = 'cuda'
NODE = 0 # 1 OR 2
FILE_PATH = f'./out/model_{NODE}.pth'
def load_dataset():

  torch.manual_seed(SEED)

  print("Loading dataset...")
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  batch_size = BATCH_SIZE

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  """trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)"""
  
  #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, 
   #                        worker_init_fn=lambda _: torch.manual_seed(seed))

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)


  train_size = int(0.8 * len(trainset))  # Adjust the split ratio as needed
  val_size = len(trainset) - train_size
  train_dataset, _ = torch.utils.data.random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  print(f'Length of trainset : {len(trainset)}')

  

  """classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')"""
  return trainloader, testloader#, classes




def main(arg):
  #print(f"Node is {arg}")
  NODE = int(arg[0])
  print(f'NODE: {NODE}')
  trainloader, testloader = load_dataset()
  model = ResNet18().to('cuda')
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  c=0
  pbar0 = tqdm(range(NUM_EPOCHS))
  for epoch in pbar0:
    if NODE == 1 and c != 0:
       os.remove('./central_ready.b')
       c+=1
       
    model.load_state_dict(torch.load('out/model.pth'))
    running_loss = 0.0
    pbar = tqdm(enumerate(trainloader, 0),leave=False)
    for i, data in pbar:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Move to gpu
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # zero the parameter gradients
        optimizer.zero_grad()



        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        pbar.set_postfix({'Running Loss': running_loss / (pbar.n + 1)})  # Average loss per iteration
        pbar.update()

    pbar0.set_postfix({'Epoch Running Loss': running_loss / (pbar.n + 1)})  # Average loss per iteration
    pbar0.update()

    params = []
    for _, param in model.named_parameters():
        params.append(param)

    file_name = f'./weights_{NODE}.pickle'
    file = open(file_name, 'wb')
    pickle.dump(params,file)
    file.close()

    file_name = f'ready{NODE}.b'
    file = open(file_name,'wb')
    pickle.dump('ready',file)
    file.close()
    
    print("Waiting for central node")
    while not os.path.exists("./central_ready.b"):
          pass
    



if __name__ == '__main__':
  main(sys.argv[1:])



