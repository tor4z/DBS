import torch
from torch import optim
from model import Classifier
from tqdm import tqdm
import metrics
import pandas as pd


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.model = Classifier(opt).cuda()
        self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                        step_size=opt.step_size, gamma=opt.gamma)

        self.epoch = 0
        self.best_acc = 0
        self.test_dl = None

    def step(self, data, label):
        data = data.cuda()
        label = label.cuda()

        self.model.train()

        self.optimizer.zero_grad()
        loss, pred = self.model(data, label)
        loss.backward()
        self.optimizer.step()
        return loss.detach(), pred
    
    def train_epoch(self, train_dl):
        iterator = tqdm(train_dl, leave=True, dynamic_ncols=True, ascii=True)
        iter_len = len(iterator)
        losses = []
        for i, (_, data, label) in enumerate(iterator):
            self.global_steps = (self.epoch * iter_len) + i
            iterator.set_description(f'train:[{self.epoch}/{self.opt.epochs}|{self.global_steps}]')
            loss, _ = self.step(data, label)

            losses.append(loss)
            # if self.global_steps % self.opt.sum_freq == 0:
            #     print(f'Train Loss {loss.item()}')
        return torch.tensor(losses).mean()
    
    def train(self, train_dl, validate_dl, test_dl):
        self.set_test_data(test_dl)
        while self.epoch < self.opt.epochs:
            loss = self.train_epoch(train_dl)
            print(f'Train Loss: {loss.item()}')
            self.validate(validate_dl)
            self.epoch += 1
            self.scheduler.step()

    def set_test_data(self, test_dl):
        self.test_dl = test_dl

    def test(self):
        test_dl = self.test_dl
        if test_dl is None:
            raise RuntimeError('test dataload is None.')
        iterator = tqdm(test_dl, leave=True, dynamic_ncols=True)
        itre_len = len(iterator)
        result = []

        with torch.no_grad():
            for i, (index, data, _) in enumerate(iterator):
                iterator.set_description(f'test:')
                _, pred = self.validate_step(data, None)
                result.append([index + 1, pred.item()])
        self.write_result(np.array(result))
        
    def validate(self, validate_dl):
        iterator = tqdm(validate_dl, leave=True, dynamic_ncols=True, ascii=True)
        iter_len = len(iterator)
        preds = []
        labels = []

        with torch.no_grad():
            for i, (_, data, label) in enumerate(iterator):
                iterator.set_description(f'validate:[{self.epoch}/{self.opt.epochs}|{self.global_steps}]')
                _, pred = self.validate_step(data, label)
                label = label.cuda()
                preds.append(pred)
                labels.append(label)
        
        pred = torch.cat(preds, dim=0)
        label = torch.cat(labels, dim=0)
        curr_acc = metrics.acc(pred, label)
        if curr_acc > self.best_acc:
            self.best_acc = curr_acc
        print(f'Current ACC {curr_acc}')
        print(f'Best ACC {self.best_acc}')

    def validate_step(self, data, label=None):
        data = data.cuda()
        if label is not None:
            label = label.cuda()
        self.model.eval()

        loss, pred = self.model(data, label)

        return loss, pred

    def write_result(self, data):
        df = pd.DataFrame({'ImageId': data[:, 0],
                           'Label': data[:, 1]})
        df.to_csv(self.opt.out_file, index=False)
