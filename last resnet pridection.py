#!/usr/bin/env python
# coding: utf-8

# Let us start by importing the libraries:

# In[ ]:


import os
import torch
import torchvision
from IPython import get_ipython
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


# Let us see the classes present in the dataset:

# In[ ]:


data_dir = 'C:\\Users\\Acer\\OneDrive\\Desktop\\New folder (5)\\Garbage Classification dataset\\Garbage classification\\Garbage classification'

classes = os.listdir(data_dir)
print(classes)


# ## Transformations:

# Now, let's apply transformations to the dataset and import it for use.

# In[ ]:


from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)


# Let's create a helper function to see the image and its corresponding label:

# In[ ]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(img.permute(1, 2, 0))
plt.show()  # Make sure this line is present to display the plot

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


# In[ ]:


img, label = dataset[12]
show_sample(img, label)


# # Loading and Splitting Data:

# In[ ]:


random_seed = 42
torch.manual_seed(random_seed)


# We'll split the dataset into training, validation and test sets:

# In[ ]:


train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)


# In[ ]:


from torch.utils.data.dataloader import DataLoader
batch_size = 32


# Now, we'll create training and validation dataloaders using `DataLoader`.

# In[ ]:


train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)


# This is a helper function to visualize batches:

# In[ ]:


from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break


# In[ ]:


show_batch(train_dl)


# # Model Base:

# Let's create the model base:

# In[ ]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))


# We'll be using ResNet50 for classifying images:

# In[ ]:


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, len(dataset.classes))
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Ensure the model is in evaluation mode after loading weights
    return model

# Instantiate your model
model = ResNet()


# ## Porting to GPU:

# GPUs tend to perform faster calculations than CPU. Let's take this advantage and use GPU for computation:

# In[ ]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[ ]:


# device = get_default_device()
# device


# In[ ]:


# train_dl = DeviceDataLoader(train_dl, device)
# val_dl = DeviceDataLoader(val_dl, device)
# to_device(model, device)


# # Training the Model:

# This is the function for fitting the model.

# In[ ]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Let's start training the model:

# # Visualizing Predictions:

# In[ ]:


def predict_image(img, model, model_filepath):
    # Load the model if it's not already loaded
    if not hasattr(predict_image, 'model_loaded'):
        # Instantiate a new model with the same architecture
        predict_image.model_loaded = ResNet()
        # Load the saved model weights
        predict_image.model_loaded = load_model(predict_image.model_loaded, model_filepath)

    # Ensure consistency between the model device and the input tensor device
    device = next(predict_image.model_loaded.parameters()).device
    # Move the input tensor to the same device as the model
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

# Example usage:
# Assuming you have trained your model and saved its weights
model_filepath = 'resnet_model_weights.pth'  # Provide the correct filepath
predicted_class = predict_image(img, model, model_filepath)


# Let us see the model's predictions on the test dataset:

# In[ ]:


model_loaded = ResNet()
model= load_model(model_loaded, 'resnet_model_weights.pth')


# In[ ]:


img, label = test_ds[17]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, model_filepath))


# In[ ]:


from PIL import Image
from pathlib import Path

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, model, model_filepath) + ".")
    
predict_external_image('WhatsApp Image 2024-03-08 at 11.07.38 PM.jpeg')


# In[ ]:


import cv2
import PySimpleGUI as sg
from PIL import Image, ImageTk
from pathlib import Path

def predict_external_image(model, model_filepath):
    # Open camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Display the captured image
        window['image'].update(data=cv2.imencode('.png', frame)[1].tobytes())
        
        # Perform prediction
        example_image = transformations(image)
        predicted_class = predict_image(example_image, model, model_filepath)
        window['output'].update("Predicted class: " + predicted_class)
        
        # Check for user input to exit
        event, values = window.read(timeout=20)  # Timeout for event polling set to 20ms
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
    
    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

# Create the PySimpleGUI layout
layout = [
    [sg.Image(filename='', key='image')],
    [sg.Text(size=(40, 1), key='output')],
    [sg.Button('Exit')]
]

# Create the window
window = sg.Window('Image Capture', layout, finalize=True)

# Assuming you have already trained your model and saved its weights
model_filepath = 'resnet_model_weights.pth'  # Provide the correct filepath
predict_external_image(model, model_filepath)

# Close the window
window.close()


# In[ ]:


img, label = test_ds[23]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# In[ ]:


img, label = test_ds[51]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))


# # Predicting External Images:

# Let's now test with external images.
# 
# I'll use `urllib` for downloading external images.

# In[ ]:


import urllib.request
urllib.request.urlretrieve("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fengage.vic.gov.au%2Fapplication%2Ffiles%2F1415%2F0596%2F9236%2FDSC_0026.JPG&f=1&nofb=1", "plastic.jpg")
urllib.request.urlretrieve("https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fi.ebayimg.com%2Fimages%2Fi%2F291536274730-0-1%2Fs-l1000.jpg&f=1&nofb=1", "cardboard.jpg")    
urllib.request.urlretrieve("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.2F0uH6BguQMctAYEJ-s-1gHaHb%26pid%3DApi&f=1", "cans.jpg") 
urllib.request.urlretrieve("https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftinytrashcan.com%2Fwp-content%2Fuploads%2F2018%2F08%2Ftiny-trash-can-bulk-wine-bottle.jpg&f=1&nofb=1", "wine-trash.jpg")
urllib.request.urlretrieve("http://ourauckland.aucklandcouncil.govt.nz/media/7418/38-94320.jpg", "paper-trash.jpg")


# Let us load the model. You can load an external pre-trained model too!

# In[ ]:


loaded_model = model


# This function takes the image's name and prints the predictions:

# In[ ]:


from PIL import Image
from pathlib import Path

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")


# In[ ]:


from PIL import Image
from pathlib import Path

def predict_external_image(image_name):
    image = Image.open(Path('./' + image_name))

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, loaded_model) + ".")
    
predict_external_image('cans.jpg')

