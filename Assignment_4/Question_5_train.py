import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets,transforms
from torch import nn,optim
import torch.utils.data


#########################################################################################
#####transform.ToTensor will first transform the image into image and
####transform.normalise will normalise the data set and variance
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

trainset=datasets.MNIST('/home/rohitk/Desktop/MLSP/a4',download=True,train=True,transform=transform)
valset=datasets.MNIST('/home/rohitk/Desktop/MLSP/a4',download=True,train=False,transform=transform)

###########converting the images into the batches
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader) #####converting into iteration type and using next you are accessing its different batches
images,labels=dataiter.next()
######getting the shape of each batch where each batch size is 64 in which 28*28 images are there
print(images.shape)
print(labels.shape)

input_size=784
hidden_size=512
output_size=10
######defining the model
model=nn.Sequential(nn.Linear(input_size,hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size,hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size,output_size),
                    nn.LogSoftmax(dim=1))
####print(model)
criterion=nn.NLLLoss()
images,labels=next(iter(trainloader))
images = images.view(images.shape[0], -1)

####the logps size is 64*10 i.e the output of the mmodel is conditional density
#####that given the image what are the conditional density of being belong to thant class
logps = model(images) ####log probabilites
loss = criterion(logps,labels)

print(torch.exp(logps[0,:]))

print(loss)
#########Actual Training that will occur

optimizer=optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
time0=time()
epochs=15
for e in range(epochs):
    running_loss=0
    for images,labels in trainloader:
        #####flatten the image first
        images=images.view(images.shape[0],-1)
        ####training pass
        ####first make the grad zero
        optimizer.zero_grad()

        output=model(images)
        loss=criterion(output,labels)

        ######Apply the bitchy backpapogation

        loss.backward()
        ####optimising the weights

        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))

print("\nTraining Time (in minutes) =",(time()-time0)/60)

correct_count,all_count=0,0
for images,labels in valloader:
    for i in range(len(labels)):
        img=images[i].view(1,784)
        with torch.no_grad():
            logps=model(img)

        ps=torch.exp(logps)
        probab=list(ps.numpy()[0])
        pred_labels = probab.index(max(probab))
        true_label=labels.numpy()[i]
        if (true_label==pred_labels):
            correct_count += 1
        all_count +=1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model, '/home/rohitk/Desktop/MLSP/a4/my_mnist_model.pt')




