import torchvision.models as models
import torch
from torch import nn
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-1]
resnet50 = nn.Sequential(*modules)
resnet50 = resnet50.cuda()
for p in resnet50.parameters():
    p.requires_grad = False

class Task1Model(nn.Module):
    def __init__(self):
        super(Task1Model, self).__init__()

        self.classify = nn.Sequential(
            nn.Linear(2048*2*4, 2048),
            nn.BatchNorm1d(2048, momentum=0.5),
            nn.ReLU(),
			# nn.Linear(4096, 1024),
			# nn.BatchNorm1d(1024,momentum=0.5),
			# nn.ReLU(),
			nn.Linear(2048, 256),
			nn.BatchNorm1d(256, momentum=0.5),
			nn.ReLU(),
			nn.Linear(256,11),
			nn.BatchNorm1d(11, momentum=0.5),
			nn.Softmax()
            # nn.LogSoftmax()
        )

    def forward(self, x):
        return self.classify(x)

class Task1ModelDeep(nn.Module):
    def __init__(self):
        super(Task1ModelDeep, self).__init__()

        self.classify = nn.Sequential(
            nn.Linear(2048*2*4, 4096),
            nn.BatchNorm1d(4096, momentum=0.5),
            nn.ReLU(),
			nn.Linear(4096, 2048),
			nn.BatchNorm1d(2048,momentum=0.5),
			nn.ReLU(),
			nn.Linear(2048, 256),
			nn.BatchNorm1d(256, momentum=0.5),
			nn.ReLU(),
			nn.Linear(256,11),
			nn.BatchNorm1d(11, momentum=0.5),
			# nn.Softmax()
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.classify(x)

class Task2Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Task2Model, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).cuda()

class Task2Model_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Task2Model_GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input = F.relu(input)
        output, hidden = self.gru(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size).cuda()

class Task2Model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Task2Model_LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.hidden = self.init_hidden()

    def forward(self, input, hidden):
        # input = F.relu(input)
        output, hidden = self.lstm(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size).cuda(), torch.zeros(1, 1, self.hidden_size).cuda())

class Task2Model_LSTM_2layer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(Task2Model_LSTM_2layer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.hidden = self.init_hidden()

    def forward(self, input, hidden):
        # input = F.relu(input)
        output, hidden = self.lstm(input, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return (torch.zeros(self.num_layer, 1, self.hidden_size).cuda(), torch.zeros(self.num_layer, 1, self.hidden_size).cuda())

if __name__ == '__main__':
    model = Task1Model()
    # print(model)
    summary(model.cuda(), [(3, 240, 320), (3, 240, 320)])
