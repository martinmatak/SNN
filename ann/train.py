import ast
import torch
import numpy as np
from torch.utils.data import Dataset
from model import FeedforwardNeuralNetModel
import sys
sys.path.insert(1, '../data_loader')
from force_data_reader import ForceDataReader
from statistics import mean

torch.manual_seed(7710)
np.random.seed(7710)

DATA_FOLDER = "../dataset/"
TRESHOLD = 3.3

class TorchDatasetWrapper(Dataset):
    def __init__(self, forceDataReader):
        self.forceDataReader = forceDataReader

    def __len__(self):
        return len(self.forceDataReader.dataset)

    def __getitem__(self, idx):
        _, tared, force_3d = self.forceDataReader.get_batch_data(idx, idx+1)
        force = np.linalg.norm(force_3d)
        if force > TRESHOLD:
            label = torch.as_tensor([1.0])
        else:
            label = torch.as_tensor([0.0])

        return torch.from_numpy(tared), label

input_dim = 19 # electrodes
hidden_dim = 100
output_dim = 1

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

# use gpu if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function that is optimized
loss_fn = torch.nn.BCEWithLogitsLoss() # applies sigmoid function to the last layer

# optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# load dataset
train_pickle = DATA_FOLDER+'train'
train_dataset = TorchDatasetWrapper(ForceDataReader(train_pickle))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

test_pickle = DATA_FOLDER+'test'
test_dataset = TorchDatasetWrapper(ForceDataReader(test_pickle))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))

def binary_acc(y_pred, y_test):
    # computes accuracy
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

# train model
num_epochs = 50
model.double()
model.train()
previous_test_acc = None

for epoch in range(num_epochs):
    epoch_losses = []
    epoch_accuracies = []

    for x,y in train_loader:
        # use GPU
        x = x.requires_grad_().to(device)
        y = y.to(device)

        # one channel only (gotta love pytorch)
        y = y.unsqueeze(1)

        # reset gradients 
        optimizer.zero_grad()

        # predict output
        output = model(x.double())

        # compare with the desired output i.e. true label
        loss = loss_fn(output, y)

        # compute gradients
        loss.backward()

        # update weights using newly computed gradients
        optimizer.step()

        # accumulate loss and accuracy for this epoch
        epoch_losses.append(loss.item())
        epoch_accuracies.append(binary_acc(output, y).item())

    epoch_loss = mean(epoch_losses) 
    epoch_acc = mean(epoch_accuracies) 
    print(f'Epoch {epoch+0:03}: | Loss: {epoch_loss:.5f} | Acc: {epoch_acc:.3f}')

    if epoch % 5 == 0:
        acc = 0.0
        # compute validation loss to see whether to stop training
        with torch.no_grad():
            for x,y in test_loader:
                x = x.to(device)
                y = y.to(device)
                y = y.unsqueeze(1)

                y_pred = model(x.double())
                acc += binary_acc(y_pred, y).item()
            test_accuracy = acc/len(test_loader)
            print("test accuracy: ", test_accuracy)
            if previous_test_acc is None or test_accuracy > previous_test_acc:
                previous_test_acc = test_accuracy

            # stop training when test accuracy starts decreasing
            if previous_test_acc > test_accuracy:
                print("stopping because there's no improvement in training anymore")
                break

# evaluate the model
model.eval() 
eval_pickle = DATA_FOLDER+'eval'
eval_dataset = TorchDatasetWrapper(ForceDataReader(eval_pickle))
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=len(eval_dataset))

with torch.no_grad():
    for x,y in eval_loader:
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze(1)

        y_pred = model(x.double())
        acc = binary_acc(y_pred, y).item()
        print("final accuracy on the evaluation dataset: ", acc)

