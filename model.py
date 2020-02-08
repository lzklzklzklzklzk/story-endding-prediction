import torch
from transformers import BertModel
from torch.utils.data import dataset
from torch.utils.data import dataloader
import torch.nn as nn
import numpy as np
import torch.optim as optim
import config


class Dataset(dataset.Dataset):
    def __init__(self, stories, labels):
        super().__init__()
        self.stories = stories
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        story = torch.tensor(self.stories[index], dtype=torch.long)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return story, label


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_layer = BertModel.from_pretrained('bert-large-cased')
        self.linear = nn.Linear(1024, 2)
        self.softmax = nn.Softmax()

    def forward(self, story):
        bert_output = self.bert_layer(story)
        #linear_output = self.linear(bert_output[1])
        linear_output = self.linear(torch.mean(
            bert_output[0], dim=1, dtype=torch.float32))
        output = self.softmax(linear_output)
        return output


def get_accuracy_from_logits(logits, label):
    l = len(label)
    c = 0
    pred_output = torch.argmax(logits, dim=1)
    for i in range(l):
        if pred_output[i] == label[i]:
            c += 1
    return float(c)/l


def get_accuracy(pred, label):
    l = len(label)
    c = 0
    for i in range(l):
        if pred[i] == label[i]:
            c += 1
    return float(c) / l


def valid(model, criterion, val_loader):
    model.eval()

    with torch.no_grad():
        losses = torch.tensor([], dtype=torch.float32)
        pred = torch.tensor([], dtype=torch.long).cuda()
        truth = torch.tensor([], dtype=torch.long).cuda()
        total = 0
        for it, (seq, label) in enumerate(val_loader):
            seq, label = seq.cuda(), label.cuda()

            logits = model(seq)

            total = total + len(label)

            loss = criterion(logits, label)

            losses = torch.cat((losses, torch.tensor([loss])))

            pred = torch.cat((pred, torch.argmax(logits, dim=1)))
            truth = torch.cat((truth, label))

        acc = get_accuracy(pred, truth)
        loss = torch.sum(losses) / total
        print("Validation complete. Loss : {} Accuracy : {}\n".format(loss, acc))

        return pred


def train(model, criterion, opti, train_loader, val_loader):

    for epoch in range(config.epochs):

        model.train()
        for it, (seq, label) in enumerate(train_loader):
            # clear gradients
            opti.zero_grad()
            # convert to cuda tensors
            seq, label = seq.cuda(), label.cuda()

            logits = model(seq)

            # loss
            loss = criterion(logits, label)

            # backpropagation
            loss.backward()

            opti.step()

            acc = get_accuracy_from_logits(logits, label)
            print("Iteration {} of  epoch {} complete. Loss : {} Accuracy : {}".format(
                it+1, epoch+1, loss.item(), acc))

        valid(model, criterion, val_loader)


def test(model, test_loader):
    pred = []
    model.eval()
    with torch.no_grad():
        for it, (seq, label) in enumerate(test_loader):
            seq, label = seq.cuda(), label.cuda()

            logits = model(seq)

            length = len(label) // 2

            for i in range(length):
                if logits[i][1] > logits[i + 1][1]:
                    pred.append(1)
                else:
                    pred.append(2)

    return pred


if __name__ == '__main__':
    stories = np.load('drive/My Drive/sct/data/train_data.npy')
    labels = np.load('drive/My Drive/sct/data/train_labels.npy')

    test_stories = np.load('drive/My Drive/sct/data/test_data.npy')
    test_labels = np.zeros((len(test_stories,)), dtype=np.int)

    stories_from_train = np.load(
        'drive/My Drive/sct/data/val_from_train_data.npy')
    labels_from_train = np.load(
        'drive/My Drive/sct/data/val_from_train_labels.npy')

    length = len(stories)
    train_len = int(length * 0.8)

    train_stories = stories[:train_len]
    val_stories = stories[train_len:]

    train_labels = labels[:train_len]
    val_labels = labels[train_len:]

    train_data = Dataset(train_stories, train_labels)
    val_data = Dataset(val_stories, val_labels)
    test_data = Dataset(test_stories, test_labels)
    val_from_train_data = Dataset(stories_from_train, labels_from_train)

    train_loader = dataloader.DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = dataloader.DataLoader(
        val_data, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=False)
    test_loader = dataloader.DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=False)
    val_from_train_loader = dataloader.DataLoader(
        val_from_train_data, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    print('initializing model')
    model = Model()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    opti = optim.Adam(model.parameters(), lr=config.learning_rate)
    print('start training')
    train(model, criterion, opti, train_loader, val_loader)
  
    print('validation on the official validation set')
    validation_data = Dataset(stories, labels)
    validation_loader = dataloader.DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    validation_pred = valid(model, criterion, validation_loader)

    with open('drive/My Drive/sct/data/validation_results.txt', 'w') as f:
        l = len(validation_pred)

        for i in range(l):
            f.write(str(validation_pred[i].item()) + '\n')

    print('validation on data from the official training set')
    val_from_train_pred = valid(model, criterion, val_from_train_loader)

    print('generating test result')
    pred = test(model, test_loader)

    with open('drive/My Drive/sct/data/test_results.txt', 'w') as f:
        l = len(pred)

        for i in range(l):
            f.write(str(pred[i]) + '\n')

    print('saving model')
    torch.save(model, 'drive/My Drive/sct/data/model.pkl')
