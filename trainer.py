import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        ''' Get the current learning rate '''
        return self._optimizer.param_groups[0]['lr']



class TransformerTrainer_masked:
    '''Trainer for fixed masked sequences'''
    def __init__(
        self, 
        model, 
        train_dataloader, 
        d,
        betas,
        warmup_steps,
        eps,
        log_freq=10,
        device="cuda:0",
        test_dataloader=None, 
        val_dataloader=None,
    ):
        self.device = device
        self.model = model.to(device)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.val_data = val_dataloader 
        self.epochs_no_improve = 0
        self.optimizer = Adam(self.model.parameters(), betas=betas, eps=eps)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.log_freq = log_freq
        self.scheduler = ScheduledOptim(
            self.optimizer, d, n_warmup_steps=warmup_steps
        )

        self.results = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'lr'])

        print("Total Parameters:", sum(p.numel() for p in self.model.parameters()))
        print("\n")

    def validate(self, epoch):
        if self.val_data is not None:
            avg_loss, avg_accuracy = self.iteration(epoch, self.val_data, train=False)
            self.results.loc[epoch, ['val_loss', 'val_accuracy']] = [avg_loss, avg_accuracy]

    def train(self, epoch):
        avg_loss, avg_accuracy = self.iteration(epoch, self.train_data, train=True)
        self.loss = avg_loss
        self.results.loc[epoch, ['epoch', 'train_loss', 'train_accuracy', 'lr']] = [epoch, avg_loss, avg_accuracy, self.scheduler.get_lr()]
        self.validate(epoch)
        return False  # continue training 


    def compute_accuracy(self, input_ids, labels, logits):
        mask_indices = (input_ids == 20)  # 20 = '[MASK]' number
        probabilities = F.softmax(logits, dim=-1)
        top5_probabilities, top5_indices = torch.topk(probabilities, 5, dim=-1)

        predicted_number = top5_indices[mask_indices][:, 0].tolist()
        original_number = labels[mask_indices].tolist()
        train_error = [int(number == pred_number) for number, pred_number in zip(original_number, predicted_number)]

        mean_train_error = np.mean(train_error)

        return mean_train_error    
    
    
    def iteration(self, epoch, data_loader, train=True):
        avg_loss = 0.0
        avg_accuracy = 0.0

        mode = "train" if train else "test" if data_loader == self.test_data else "val"

        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP_%s:%d" % (mode, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )


        for i, batch in data_iter:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids)

            mask_token_id = 20  # 20 = '[MASK]' number
            mask = (input_ids == mask_token_id)
            logits_masked = logits[mask]
            labels_masked = labels[mask]

            loss = self.criterion(logits_masked, labels_masked)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.scheduler.step_and_update_lr()

            avg_accuracy += self.compute_accuracy(input_ids, labels, logits)
            avg_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "loss": avg_loss / (i + 1),
                "accu": avg_accuracy / (i + 1),
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print(
            f"EP{epoch}, {mode}: avg loss={avg_loss / len(data_iter)}, lr = {self.scheduler.get_lr()}, avg accuracy = {avg_accuracy / len(data_iter)}\n"
        )


        if train: 
            self.accuracy_train = avg_accuracy / len(data_iter)
        else: 
            self.accuracy_test = avg_accuracy / len(data_iter)

        return avg_loss / len(data_iter), avg_accuracy / len(data_iter)
