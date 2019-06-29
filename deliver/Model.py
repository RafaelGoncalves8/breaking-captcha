#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import random
import gc

import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

import cv2
from imutils import paths
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

os.sys.path.append('../src')
from helpers import resize_to_fit


# In[25]:


data_dir = os.path.abspath(os.path.relpath('../data'))
image_dir = os.path.abspath(os.path.relpath('../doc/images'))


# In[26]:


CAPTCHA_IMAGES_FOLDER = "../data/samples"

# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(CAPTCHA_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Grab the labels
    label = image_file.split(os.path.sep)[-1].split('.')[-2]

    # Add the image and it's label to our training data
    data.append(image)
    labels.append(label)


# In[27]:


(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)


# In[28]:


def create_images(data, label):
    # Otsu threashold
    data_pre = []
    for e in data:
        ret, th = cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(th, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        data_pre.append(erosion)
    
    # K-means
    data_pts = []
    for e in data_pre:
        data_pts.append(np.where(e == 0))
    data_pts = np.array(data_pts)
    
    X = []
    thres = 3
    for e in data_pts:
        x = (np.vstack((e[1],np.flip(e[0])))).T
        l = []
        # Discard columns with less than thres points
        for i in range(200):
            if len(x[x[:,0] == i]) > thres:
                for f in x[x[:,0] == i]:
                    l.append(f)
        x = np.array(l)
        X.append(x)
    X = np.array(X)
    
    # Projection in x-axis
    X_proj = [x[:,0].reshape(-1,1) for x in X]
    
    # Find clusters in projected data
    y_kmeans_proj = []
    centers_kmeans_proj = []
    for i, x in enumerate(X_proj):
        kmeans = KMeans(n_clusters=5)#, init=np.array([(i*200/6.0, 25) for i in range(1,6)]))
        kmeans.fit(x)
        centers_kmeans_proj.append(kmeans.cluster_centers_)
        y_kmeans_proj.append(kmeans.predict(x))

    centers = [np.sort(e, axis=0) for e in centers_kmeans_proj]
    data_chars = []
    for i, e in enumerate(data_pre):
        chars = []
        for j in range(5):
            chars.append(e[:,int(centers[i][j]-21):int(centers[i][j]+21)])
        data_chars.append(chars)
        
    return data_chars


# In[30]:


letters_train_dir = '../data/letters/train'

data_chars = create_images(X_train, y_train)

if not(os.path.isdir(''.join((letters_train_dir)))):
    os.mkdir(''.join((letters_train_dir)))

for i,e in enumerate(data_chars):
    for j in range(5):
        if not(os.path.isdir(''.join((letters_train_dir,'/',y_train[i],'/')))):
            os.mkdir(''.join((letters_train_dir,'/',y_train[i],'/')))
        cv2.imwrite(''.join((letters_train_dir,'/',y_train[i],'/',y_train[i][j],'.png')),e[j])


# In[31]:


letters_test_dir = '../data/letters/test'

data_chars_test = create_images(X_test, y_test)

if not(os.path.isdir(''.join((letters_test_dir)))):
    os.mkdir(''.join((letters_test_dir)))

for i,e in enumerate(data_chars_test):
    for j in range(5):
        if not(os.path.isdir(''.join((letters_test_dir,'/',y_test[i],'/')))):
            os.mkdir(''.join((letters_test_dir,'/',y_test[i],'/')))
        cv2.imwrite(''.join((letters_test_dir,'/',y_test[i],'/',y_test[i][j],'.png')),e[j])


# In[32]:


LETTER_IMAGES_FOLDER = letters_train_dir

# initialize the data and labels
data_l_train = []
labels_l_train = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 28x28 pixel box
    image = resize_to_fit(image, 28, 28)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2].split('.')[-2]

    # Add the letter image and it's label to our training data
    data_l_train.append(image)
    labels_l_train.append(label)


# In[33]:


LETTER_IMAGES_FOLDER = letters_test_dir

# initialize the data and labels
data_l_train = []
labels_l_train = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 28x28 pixel box
    image = resize_to_fit(image, 28, 28)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-1].split('.')[-2]

    # Add the letter image and it's label to our training data
    data_l_train.append(image)
    labels_l_train.append(label)


# In[36]:


# scale the raw pixel intensities to the range [0, 1] (this improves training)
X_l_train = np.array(data_l_train, dtype="float") / 255.0
y_l_test = np.array(data_l_test, dtype="float") / 255.0


# In[22]:


# Convert the labels (letters) into one-hot encodings that Keras can work with
le = LabelEncoder().fit(np.array(labels_l_train))
y_train = le.transform(np.array(y_train)
y_test = le.transform(y_test)


# In[23]:


batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
n_epochs = 15
log_interval = 10


# In[24]:


X_train_t = (torch.from_numpy(X_train).float().transpose(1,3)).transpose(2,3)
y_train_t = torch.from_numpy(y_train).long()

train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=round(batch_size_train), shuffle=True)


# In[25]:


X_test_t = (torch.from_numpy(X_test).float().transpose(1,3)).transpose(2,3)
y_test_t = torch.from_numpy(y_test).long()

test_data = torch.utils.data.TensorDataset(X_test_t, y_test_t)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# In[ ]:





# In[ ]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 120)
        self.fc2 = nn.Linear(120, 32)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return F.log_softmax(x, dim=0)


# In[ ]:


def train(epoch, v=True):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if v:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
        torch.save(net.state_dict(), 'model.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')


# In[ ]:


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# In[ ]:


def model(data, labels, learning_rate=0.01, n_epochs=5, ):
    create_images(data, labels)
    
    
    
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    for epoch in range(1, n_epochs + 1):
        train(epoch)


# In[1]:


LETTER_IMAGES_FOLDER = '../data/letters'


# In[20]:


# initialize the data and labels
data = []
labels = []

# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the letter so it fits in a 28x28 pixel box
    image = resize_to_fit(image, 28, 28)

    # Add a third channel dimension to the image to make Keras happy
    image = np.expand_dims(image, axis=2)

    # Grab the name of the letter based on the folder it was in
    label = image_file.split(os.path.sep)[-2]

    # Add the letter image and it's label to our training data
    data.append(image)
    labels.append(label)


# In[21]:


# scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# In[22]:


# Split the training data into separate train and test sets
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
le = LabelEncoder().fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)


# In[23]:


batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
n_epochs = 15
log_interval = 10


# In[24]:


X_train_t = (torch.from_numpy(X_train).float().transpose(1,3)).transpose(2,3)
y_train_t = torch.from_numpy(y_train).long()

train_data = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=round(batch_size_train), shuffle=True)


# In[25]:


X_test_t = (torch.from_numpy(X_test).float().transpose(1,3)).transpose(2,3)
y_test_t = torch.from_numpy(y_test).long()

test_data = torch.utils.data.TensorDataset(X_test_t, y_test_t)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# In[ ]:



