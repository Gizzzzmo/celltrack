import torch
import torch.nn.functional as F
from setup import device


def simple_nn(X, y, train_len, batch_size=50, early_stopping_patience=500):
    """trains a simple convolutional neural network for binary classification on the given samples X, and ground truths y"""
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, 2)
            self.bn1 = torch.nn.BatchNorm2d(32)
            self.conv2 = torch.nn.Conv2d(32, 32, 3)
            self.bn2 = torch.nn.BatchNorm2d(32)
            self.conv3 = torch.nn.Conv2d(32, 64, 2)
            self.bn3 = torch.nn.BatchNorm2d(64)
            self.conv4 = torch.nn.Conv2d(64, 64, 2)
            self.bn4 = torch.nn.BatchNorm2d(64)
            self.conv5 = torch.nn.Conv2d(64, 128, 2)
            self.bn5 = torch.nn.BatchNorm2d(128)
            self.fc2 = torch.nn.Linear(128, 128)
            self.fc1 = torch.nn.Linear(128, 2)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = torch.mean(x, dim=(2, 3)).view(-1, 128)
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc1(x), dim=1)
            return x
    
    net = Net().cuda()
    # binary cross entropy
    criterion = torch.nn.CrossEntropyLoss()
    # stochastic gradient descent
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)


    best_loss = float('inf')
    cutting_counter = 0
    best_val_loss = float('inf')
    stopping_counter = 0
    X = X.reshape(len(X), 1, 24, 24)

    # split data into training and validation set
    train_x = torch.from_numpy(X[:train_len]).type(torch.FloatTensor)
    val_x = torch.from_numpy(X[train_len:]).type(torch.FloatTensor)
    train_y = torch.from_numpy(y[:train_len]).type(torch.LongTensor).to(device)
    val_y = torch.from_numpy(y[train_len:]).type(torch.LongTensor).to(device)

    best_accuracy = 0
    best_network = Net().cuda()
    best_network.load_state_dict(net.state_dict())

    # train for 2000 epochs or until the validation loss hasn't improved for early_stopping_patience many iteraions
    for epoch in range(2000):
        running_loss = 0.0
        for i in range(0, len(train_x), batch_size):
            # set the network to training mode
            net.train()
            # create the correct batch of data
            inputs = train_x[i:min(i+batch_size, len(train_x))].to(device)
            labels = train_y[i:min(i+batch_size, len(train_x))]
            
            optimizer.zero_grad()
            # compute the loss, backprop the gradients and update the cnn weights
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # set the network to evaluation mode and compute validation loss and accuracy
        net.eval()
        val_out = net(val_x[::10].to(device))
        pred = val_out[:, 0] < 0.5
        print('acc:', torch.sum(pred == val_y[::10]).item()/len(val_y[::10]))
        val_loss = criterion(val_out, val_y[::10])
        print('[%d, %5d] train loss: %.9f' %
            (epoch + 1, i + 1, running_loss / len(train_x)))
        print('[%d, %5d] val loss: %.9f' %
            (epoch + 1, i + 1, val_loss / len(val_x)))

        # implement early stopping
        if val_loss/len(val_x) < best_val_loss:
            best_val_loss = val_loss/len(val_x)
            stopping_counter = 0
        else:
            stopping_counter += 1
            if stopping_counter > early_stopping_patience:
                break
        
        # cut the learning rate by a factor of .1 if the training accuracy plateaus
        if (acc := torch.sum(pred == val_y[::10]).item()/len(val_y[::10])) > best_accuracy:
            print('new best!')
            best_accuracy = acc
            best_network.load_state_dict(net.state_dict())
        if running_loss / len(train_x) < best_loss:
            best_loss = running_loss / len(train_x)
            cutting_counter = 0
        else:
            cutting_counter += 1
            if cutting_counter > 5:
                cutting_counter = 0
                print('dropping learn rate')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

    # set the network to evaluation mode before returning it
    best_network.eval()

    return best_network
