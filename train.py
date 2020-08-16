import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import scipy.stats
import matplotlib.pyplot as plt
from setup import device
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def log_reg(X, y):
    clf = LogisticRegression(random_state=0).fit(X.reshape(len(X), 1), y)
    xaxis = np.arange(100)/(100 * (np.max(X)-np.min(X))) + np.min(X)
    plt.plot(X, y, 'o')
    plt.plot(xaxis, clf.predict_proba(xaxis.reshape(100, 1)))
    plt.show()

    return clf

def bayes(X, y):
    class_distributions = {}
    class_prob = {}
    classes = np.unique(y)
    for c in classes:
        dist = np.extract(y == c, X)
        class_distributions[c] = scipy.stats.norm(np.mean(dist), np.std(dist))
        class_prob[c] = len(dist)/len(X)
    
    def classify(X, h=1e-8):
        denom = 0
        seeing_x = []
        for c in class_prob:
            seeing_x.append(class_prob[c] * (class_distributions[c].cdf(X + 0.5*h) - class_distributions[c].cdf(X - 0.5*h)))
            denom += seeing_x[-1]
        seeing_x = np.stack(seeing_x)
        return classes[np.argmax(seeing_x, axis=0)]
    return classify

def knn(X, y, n_neighbors=10):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X, y)
    predictions = neigh.predict(X)
    if X.shape[1] == 1:
        plt.plot(np.extract(predictions, X), np.ones(predictions.sum()), 'o')
        plt.plot(np.extract(predictions == False, X), np.zeros(len(X) - predictions.sum()), 'ro')
    elif X.shape[1] == 2:
        print(predictions.shape, X.shape)
        plt.plot(X[predictions, 0], X[predictions, 1], 'o')
        plt.plot(X[predictions == False, 0], X[predictions == False, 1], 'ro')
        plt.plot()
        plt.plot(X[y, 0], X[y, 1], 'o')
        plt.plot(X[y == False, 0], X[y == False, 1], 'ro')
    plt.show()

    return neigh

def simple_nn(X, y, train_len, batch_size=50, early_stopping_patience=500):
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

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    best_loss = float('inf')
    cutting_counter = 0
    best_val_loss = float('inf')
    stopping_counter = 0
    X = X.reshape(len(X), 1, 24, 24)
    train_x = torch.from_numpy(X[:train_len]).type(torch.FloatTensor)
    val_x = torch.from_numpy(X[train_len:]).type(torch.FloatTensor)
    train_y = torch.from_numpy(y[:train_len]).type(torch.LongTensor).to(device)
    val_y = torch.from_numpy(y[train_len:]).type(torch.LongTensor).to(device)
    best_accuracy = 0
    best_network = Net().cuda()
    best_network.load_state_dict(net.state_dict())
    for epoch in range(2000):
        running_loss = 0.0
        for i in range(0, len(train_x), batch_size):
            net.train()
            inputs = train_x[i:min(i+batch_size, len(train_x))].to(device)
            labels = train_y[i:min(i+batch_size, len(train_x))]
            
            #print(labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            #print(net.fc1.weight.grad)
            optimizer.step()
            #print(loss.item())

            running_loss += loss.item()
        net.eval()
        val_out = net(val_x[::10].to(device))
        pred = val_out[:, 0] < 0.5
        print('acc:', torch.sum(pred == val_y[::10]).item()/len(val_y[::10]))
        val_loss = criterion(val_out, val_y[::10])
        print('[%d, %5d] train loss: %.9f' %
            (epoch + 1, i + 1, running_loss / len(train_x)))
        print('[%d, %5d] val loss: %.9f' %
            (epoch + 1, i + 1, val_loss / len(val_x)))
        if val_loss/len(val_x) < best_val_loss:
            best_val_loss = val_loss/len(val_x)
            stopping_counter = 0
        else:
            stopping_counter += 1
            if stopping_counter > early_stopping_patience:
                break

        if (acc := torch.sum(pred == val_y[::10]).item()/len(val_y[::10])) > best_accuracy:
            print('new best!')
            best_accuracy = acc
            best_network.load_state_dict(net.state_dict())
        if running_loss / len(train_x) < best_loss:
            #print('new best:', str(running_loss / len(train_x)) + ' < ' + str(best_loss))
            best_loss = running_loss / len(train_x)
            cutting_counter = 0
        else:
            cutting_counter += 1
            if cutting_counter > 5:
                cutting_counter = 0
                print('dropping learn rate')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
    best_network.eval()

    return best_network
