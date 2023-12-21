import numpy as np
import sys
from matplotlib import pyplot as plt
from Autoencoder import Autoencoder
from Siamese import SiameseNetwork, ContrastiveLoss
import torchvision
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
sys.path.append("../")

#set the device to cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch = 36
tmp_batch = 16
siamese_batch = 1
learning_rate = 0.001
max_epoch = 30

img_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/olivettie_dataset/olivetti_faces.npy')
unique_img_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/olivettie_dataset/olivetti_faces_target.npy')

## pairs for siamese network here are 500 different pair of pictures (500, 2, 64, 64)
pairs_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/olivetti_pairs.npy')
pairs_label_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/olivetti_pairs_label.npy')

new_img_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/new_data.npy')
new_unique_img_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/new_data_label.npy')

imgA_array = []
imgB_array = []
imgA_array_test = []
imgB_array_test = []

for i in range(len(pairs_array)):
    imgA_array.append(pairs_array[i][0])   
    imgB_array.append(pairs_array[i][1])      



# print(pairs_array[499][2].shape)
# print(pairs_array[0][0])
# exit()

# 2000 pairs (2000, 2, 64, 64)
test_pairs_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/test_pairs.npy')
test_pairs_label_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/test_pairs_labels.npy')

or_test_pairs_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/test_pairs.npy')
or_test_pairs_label_array = np.load('/Users/xuankeyang/Desktop/SPRING 2023/CSCI5561/Project/test_pairs_labels.npy')

# plt.imshow(or_test_pairs_array[i][0])
# plt.show()
# plt.imshow(or_test_pairs_array[i][1])
# plt.show()

# for i in range(len(test_pairs_array)):
#     imgA_array_test.append(test_pairs_array[i][0])   
#     imgB_array_test.append(test_pairs_array[i][1])

# train_img_array = []
# train_unique_img_array = []
# test_img_array = []
# test_unique_img_array = []

# for i in range(400):
#     if ((i%10!=0)):
#         train_img_array.append(img_array[i])
#         train_unique_img_array.append(unique_img_array[i])
#     else:
#         test_img_array.append(img_array[i])
#         test_unique_img_array.append(unique_img_array[i])

# train_img_array = np.asarray(train_img_array)
# test_img_array = np.asarray(test_img_array)

train_img_array = np.asarray(new_img_array)
train_unique_img_array = np.asarray(new_unique_img_array)
# test_img_array = np.asarray(test_img_array)

pair1 = []
pair1_label = []
pair2 = []
pair2_label = []
for k in range(40):
    if (k%2 != 0):
        pair1.append(train_img_array[k])
        pair1_label.append(train_unique_img_array[k])
    else:
        pair2.append(train_img_array[k])
        pair2_label.append(train_unique_img_array[k])

pair1 = np.asarray(pair1)
pair2 = np.asarray(pair2)
pair1_label = np.asarray(pair1_label)
pair2_label = np.asarray(pair2_label)

pairs = []
labels = []

for j in range(20):
    img1 = pair1[j]
    for i in range(20):
        idx = np.random.choice(pair2.shape[0], size=1, replace=False)[0]
        img2 = pair2[idx]
        label = 1 if pair1_label[j] == pair2_label[idx] else 0
        pairs.append([img1, img2])
        labels.append(label)

# Convert the pairs and labels to numpy arrays
pairs = np.array(pairs)
labels = np.array(labels)

print(pairs.shape)
np.save("new_test_pairs.npy", pairs)
np.save("new_test_pairs_labels.npy", labels)
exit()

tmp_test_array = []
tmp_test_lable = []
for j in range(10):
    tmp_test_array.append(test_img_array[j])
    tmp_test_lable.append(test_unique_img_array[j])


# print(tmp_test_lable)

class CLASSIFIER(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    

#data_train = CLASSIFIER(img_array,unique_img_array)
data_train = CLASSIFIER(train_img_array,train_unique_img_array)
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch, shuffle=True, num_workers=0)

#tmp_data_train = CLASSIFIER(tmp_test_array,tmp_test_lable)
#tmp_data_loader_train = torch.utils.data.DataLoader(dataset=tmp_data_train, batch_size=tmp_batch, shuffle=True, num_workers=0)

imgA_output = CLASSIFIER(imgA_array,pairs_label_array)
imgA_output_loader = torch.utils.data.DataLoader(dataset=imgA_output, batch_size=1, shuffle=True, num_workers=0)
imgB_output = CLASSIFIER(imgB_array,pairs_label_array)
imgB_output_loader = torch.utils.data.DataLoader(dataset=imgB_output, batch_size=1, shuffle=True, num_workers=0)

objA = enumerate(imgA_output_loader)
objB = enumerate(imgB_output_loader)

criterion = nn.MSELoss() ## loss used for autoencoder
model = Autoencoder(4096,200)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

output_arrayA = []
output_arrayB = []


training_loss, epoch = model.fit(data_loader_train, criterion, optimizer)
plt.plot(epoch, training_loss, label="Autoencoder Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig("Autoencoder_loss.png")
plt.close()

for j, (images, lables)in objA:
    output_arrayA.append(model.forward(images))
    
for j, (images, lables)in objB:
    output_arrayB.append(model.forward(images))


for i in range(len(output_arrayA)):
    pairs_array[i][0] = output_arrayA[i].detach().numpy().reshape((64, 64))
    pairs_array[i][1] = output_arrayB[i].detach().numpy().reshape((64, 64))
    

# The second part 
imgA_output_test = CLASSIFIER(imgA_array_test,test_pairs_label_array)
imgA_output_loader_test = torch.utils.data.DataLoader(dataset=imgA_output_test, batch_size=1, shuffle=False, num_workers=0)
imgB_output_test = CLASSIFIER(imgB_array_test,test_pairs_label_array)
imgB_output_loader_test = torch.utils.data.DataLoader(dataset=imgB_output_test, batch_size=1, shuffle=False, num_workers=0)

objA_test = enumerate(imgA_output_loader_test)
objB_test = enumerate(imgB_output_loader_test)

output_arrayA_test = []
output_arrayB_test = []
for j, (images, lables)in objA_test:
    output_arrayA_test.append(model.forward(images))
    
for j, (images, lables)in objB_test:
    output_arrayB_test.append(model.forward(images))


for i in range(len(output_arrayA_test)):
    test_pairs_array[i][0] = output_arrayA_test[i].detach().numpy().reshape((64, 64))
    test_pairs_array[i][1] = output_arrayB_test[i].detach().numpy().reshape((64, 64))




net = SiameseNetwork(1).to(device)
criterion_Siamese = ContrastiveLoss()
# Declare Optimizer
optimizer_Siamese = torch.optim.Adam(net.parameters(), lr=1e-3)
#train the model



def train(net, train_dataloader, max_epoch, loss_list=[]):
    for epoch in range(1,max_epoch):
        train_loss = 0
        for i, (pair, label) in enumerate(train_dataloader):
            img0, img1 = pair[0][0], pair[0][1]

            optimizer_Siamese.zero_grad()
            output1,output2 = net(img0,img1)
            loss_contrastive = criterion_Siamese(output1,output2,label)
            print("Constrastive Loss: ", loss_contrastive.item())
            train_loss += loss_contrastive.item()
            loss_contrastive.backward()
            optimizer_Siamese.step()    
        # print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
         
        
        loss_list.append(train_loss)

    return net, loss_list

class CLASSIFIER_PAIR(Dataset):
    def __init__(self, image_pair, labels):
        self.image_pair = image_pair
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.image_pair[index], self.labels[index]
    
pair_train = CLASSIFIER_PAIR(pairs_array,pairs_label_array)
pair_loader_train = torch.utils.data.DataLoader(dataset=pair_train, batch_size=siamese_batch, shuffle=True, num_workers=0)
model_siamese, loss_list = train(net, pair_loader_train, max_epoch)

index = np.arange(len(loss_list))
plt.plot(index, loss_list, label="Siamese Training Loss")
plt.xlabel("Epoch")
plt.ylabel('Training Loss')
plt.legend()
plt.savefig("Siamese_loss.png")
exit()


pair_test = CLASSIFIER_PAIR(test_pairs_array, test_pairs_label_array)
pair_loader_test = torch.utils.data.DataLoader(dataset=pair_test, batch_size=siamese_batch, shuffle=False, num_workers=0)

for i, (pair, label) in enumerate(pair_loader_test):
    img0, img1 = pair[0][0], pair[0][1]

    tmp0 = img0.detach().numpy().reshape((64, 64))
    tmp1 = img1.detach().numpy().reshape((64, 64))
    
    out1, out2 = model_siamese.forward(img0, img1)
    loss_contrastive = criterion_Siamese(out1, out2, label)
    
    count = 0
    if(label[0]==1):
        fig = plt.figure()
        fig.add_subplot(2, 2, 1)
        plt.title("loss_contrastive: "+str(float(loss_contrastive)))
        plt.imshow(or_test_pairs_array[i][0], cmap='gray')
        plt.axis('off')
        plt.title("Original first image")
        fig.add_subplot(2, 2, 2)
        plt.imshow(or_test_pairs_array[i][1], cmap='gray')
        plt.axis('off')
        plt.title("Original second image")
        fig.add_subplot(2, 2, 3)
        plt.imshow(tmp0, cmap='gray')
        plt.axis('off')
        plt.title("Restructure first image" )
        fig.add_subplot(2, 2, 4)
        plt.imshow(tmp1, cmap='gray')
        plt.axis('off')
        plt.title("Restructure second image")
        plt.show()

        count += 1

        # plt.imshow(or_test_pairs_array[i][0], cmap='gray')
        # plt.show()
        # plt.imshow(or_test_pairs_array[i][1], cmap='gray')
        # plt.show()
        # plt.imshow(tmp0)
        # plt.title("loss_contrastive: "+str(loss_contrastive))
        # plt.show()
        # plt.imshow(tmp1)
        # plt.title("loss_contrastive: "+str(loss_contrastive))
        # plt.show()
    

    
    
    print("Label: ", label, "Loss: ", loss_contrastive)
    if count == 5:
        break

print("FINISH")




# for i in range(len(output_array)):
#     tmp = output_array[i].detach().numpy().reshape((64, 64))
#     plt.imshow(tmp_test_array[output_label_array[i][0].int()], cmap='gray')
#     plt.show()
#     #print("num: "+str(i), tmp)
#     print(output_label_array[i][0].int())
#     plt.imshow(tmp, cmap='gray')
#     plt.show()