import numpy as np

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self,input_shape,max_epochs):
        super().__init__()
        self.max_epochs = max_epochs
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=input_shape, out_features=1024
        )
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=1024, out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=512, out_features=512
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=512, out_features=1024
        )
        self.decoder_output_layer = nn.Linear(
            in_features=1024, out_features=input_shape
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        activation1 = self.encoder_hidden_layer1(x)
        activation1 = torch.relu(activation1)
        activation2 = self.encoder_hidden_layer2(activation1)
        activation2 = torch.relu(activation2)
        code = self.encoder_output_layer(activation2)
        code = torch.relu(code)
        activation3 = self.decoder_hidden_layer1(code)
        activation3 = torch.relu(activation3)
        activation4 = self.decoder_hidden_layer2(activation3)
        activation4 = torch.relu(activation4)
        activation = self.decoder_output_layer(activation4)
        reconstructed = torch.relu(activation)
        return reconstructed
    
    def fit(self, train_loader, criterion, optimizer):
        train_loss_list = []
        epoch_list = []
        error_rate_list = []

        # Epoch loop
        for i in range(self.max_epochs):
            train_loss = 0
            correct = 0
            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):
                stop = False
                # Forward pass (consider the recommmended functions in homework writeup)
                outputs = self.forward(images)
                images = images.view(images.size(0), -1)
                # print(outputs.shape)
                # print(labels.shape)
               
                loss = criterion(outputs, images)
                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Track the loss and error rate
                train_loss += loss.item() 
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

            error_rate = (1 - correct/len(train_loader.dataset))
            error_rate_list.append(error_rate)

            if len(error_rate_list) >= 30:
                for k in range(1,10):
                    if abs(error_rate_list[-1] - error_rate_list[len(error_rate_list)-k]) >= 0.001:
                        stop = False
                        break
                    stop = True 
                    
            if stop == True:
                break 

            print("Epoch Loss:" + str(train_loss))

            train_loss_list.append(train_loss)
            epoch_list.append(i)
            
        return train_loss_list, epoch_list