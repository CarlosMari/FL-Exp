from config import NUM_EPOCHS
from models.ResNet import ResNet18
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import os
import pickle

FILE_PATH = f'./out/model.pth'
from config import SEED, NUM_EPOCHS, BATCH_SIZE
DEVICE = 'cuda'

def load_data():
  torch.manual_seed(SEED)
  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

  train_size = int(0.8 * len(trainset))  # Adjust the split ratio as needed
  val_size = len(trainset) - train_size
  _, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

  valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)
  
  return valloader, testloader

def main():
  model = ResNet18().to(DEVICE)
  model_state_dict = model.state_dict()
  torch.save(model_state_dict, FILE_PATH)
  criterion = nn.CrossEntropyLoss()
  valloader, testloader = load_data()

  for z in range(NUM_EPOCHS):
    print("Waiting for weights...")
    while  not (os.path.exists("ready1.b") and os.path.exists("ready2.b")):
      pass
    file = open("weights_1.pickle", 'rb')
    w_1 = pickle.load(file)  # Read the entire binary file
    file.close()

    file = open("weights_2.pickle", 'rb')
    w_2 = pickle.load(file)
    file.close()

    #w_1 = torch.tensor(w_1).to(DEVICE)
    #w_2 = torch.tensor(w_2).to(DEVICE)
    w = []
    for i in range(len(w_1)):
      w.append((w_1[i] + w_2[i])/2.0)

    assert w_1[0].shape == w[0].shape, f'{w_1[0].shape}:{w[0].shape}'

    c= 0
    for name, _ in model.named_parameters():
      # We manually assign the parameters of each layer
      model.state_dict()[name][:] = w[c]
      c+=1

    print("Saved new Parameters")
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, FILE_PATH)
    
    # We run some validation to see convergence

    model.eval()


    losses = []
    with torch.no_grad():

      for j, data in enumerate(valloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data

          # Move to gpu
          inputs = inputs.to(DEVICE)
          labels = labels.to(DEVICE)

          # forward + backward + optimize
          outputs = model(inputs)
          loss = criterion(outputs, labels)

          losses.append(loss)
          # print statistics


    avg_loss = sum(losses)/len(losses)
    print(f"Epoch: {z}: Avg Loss: {avg_loss} ")




    model.train()


    file_name = f'./central_ready.b'
    file = open(file_name,'wb')
    pickle.dump('ready',file)
    file.close()

    os.remove('ready1.b')
    os.remove('ready2.b')

  print("Testing....")

  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images = images.to(DEVICE)
          labels = labels.to(DEVICE)
          # calculate outputs by running images through the network
          outputs = model(images)
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
  main()
