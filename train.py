import argparse
import os
from typing import List, Tuple, NoReturn

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from car_dataloader import get_car_train_dataset, get_car_validation_dataset
from resnet152 import resnet152
from efficientnet.model import build_efficientnet

IMAGE_SIZE = 240
NUM_CLASSES = 196

def parse_args():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="input batch size for training (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of training epochs (default: 100)")
    parser.add_argument("--initial_lr", type=float, default=1e-4,
                        help="initial learning rate (default: 1e-4)")
    parser.add_argument("--lr_reduce_factor", type=float, default=0.1,
                        help="learning rate reduce factor (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 Regularizer (default: 1e-5)")
    parser.add_argument("--validation", action='store_true',
                        help="specify validation mode")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="the model directory (default: model)")
    parser.add_argument("--checkpoint_filename", type=str, default="checkpoint.tar",
                        help="filename of training checkpoint (default: checkpoint.tar)")
    parser.add_argument("--best_filename", type=str, default="best.tar",
                        help="filename of the best checkpoint (default: best.tar)")
    parser.add_argument("--manual_seed", type=int, default=42,
                        help="seed for torch (default: 42)")

    args = parser.parse_args()
    return args

class ModelTrainer(object):
    
    def __init__(self, args):
        super(ModelTrainer, self).__init__()
        
        self.model = resnet152(NUM_CLASSES)
        # self.model = build_efficientnet("efficientnet-b1")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.args = args
        self.model_dir = args.model_dir
        self.checkpoint_filename = os.path.join(args.model_dir, args.checkpoint_filename)
        self.best_filename = os.path.join(args.model_dir, args.best_filename)

    def _train(self,
               epoch: int,
               model: nn.Module,
               criterion: nn.Module,
               data_loader: DataLoader,
               optimizer: optim.Optimizer) -> float:
        model.train()
        
        running_loss = 0
        running_corrects = 0

        for batch_idx, (data, labels) in enumerate(data_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            logits = model(data)
            loss = criterion(logits, labels)
            corrects = self.count_corrects(logits, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * len(data)
            running_corrects += corrects.item()

            if batch_idx % 20 == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    epoch, batch_idx * len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader),
                    loss.item(),
                    corrects.item() / len(data)))

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = running_corrects / len(data_loader.dataset)
        print("====> Epoch: {} Average loss: {:.4f}, accuracy: {:.4f}".format(
            epoch, epoch_loss, epoch_accuracy))

        return epoch_loss

    def _validate(self,
                  model: nn.Module,
                  criterion: nn.Module,
                  data_loader: DataLoader) -> float:
        model.eval()
        
        running_loss = 0
        running_corrects = 0
        
        with torch.no_grad():
            for (data, labels) in data_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                logits = model(data)
                loss = criterion(logits, labels)
                corrects = self.count_corrects(logits, labels)

                # statistics
                running_loss += loss.item() * len(data)
                running_corrects += corrects.item()

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_accuracy = running_corrects / len(data_loader.dataset)
        print("====> Validation set loss: {:.4f}, accuracy: {:.4f}".format(
            epoch_loss, epoch_accuracy))

        return epoch_loss

    def load(self, load_best=False) -> Tuple[nn.Module, optim.Optimizer, object, int, int, float]:
        model = self.model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.initial_lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.args.lr_reduce_factor, verbose=True)
        
        last_state = None
        last_epoch = None
        last_loss = None
        
        best_state = None
        best_epoch = None
        best_loss = None

        if os.path.exists(self.checkpoint_filename):
            last_state = torch.load(self.checkpoint_filename)
            last_epoch = last_state["epoch"]
            last_loss = last_state["loss"]

        if os.path.exists(self.best_filename):
            best_state = torch.load(self.best_filename)
            best_epoch = best_state["epoch"]
            best_loss = best_state["loss"]

        if last_state or best_state:
            if load_best:
                if not best_state:
                    load_best = False
            else:
                if not last_state:
                    load_best = True

            if load_best:
                state = best_state
            else:
                state = last_state

            print("Reloading model at epoch {}"
                ", with test error {}".format(
                    state["epoch"],
                    state["loss"]))
            model.load_state_dict(state["state_dict"])
            optimizer.load_state_dict(state["optimizer"])
            
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])

        return model, optimizer, scheduler, last_epoch, best_epoch, best_loss

    def save_checkpoint(self,
                        state: dict,
                        is_best: str) -> NoReturn:
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        torch.save(state, self.checkpoint_filename)
        if is_best:
            torch.save(state, self.best_filename)

    def count_corrects(self,
                       logits: torch.Tensor,
                       labels: torch.Tensor) -> torch.Tensor:
        _, max_indices = torch.max(logits, 1)
        corrects = (max_indices == labels).sum()
        return corrects

    def test(self):

        model, _, _, last_epoch, best_epoch, best_loss = self.load(load_best=True)
        criterion = nn.CrossEntropyLoss()

        if not best_epoch:
            raise Exception("Best solution not found")

        validation_data = get_car_validation_dataset(IMAGE_SIZE)
        validation_data_loader = DataLoader(validation_data, batch_size=self.args.batch_size, num_workers=0)

        test_accuracy = self._validate(model, criterion, validation_data_loader)

    def train(self):
        
        model, optimizer, scheduler, last_epoch, best_epoch, best_loss = self.load()
        criterion = nn.CrossEntropyLoss()

        train_data = get_car_train_dataset(IMAGE_SIZE)
        train_data_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        validation_data = get_car_validation_dataset(IMAGE_SIZE)
        validation_data_loader = DataLoader(validation_data, batch_size=self.args.batch_size, num_workers=0)

        if not last_epoch:
            last_epoch = 0

        for epoch in range(last_epoch + 1, last_epoch + self.args.num_epochs + 1):
            self._train(epoch, model, criterion, train_data_loader, optimizer)
            validation_loss = self._validate(model, criterion, validation_data_loader)
            scheduler.step(validation_loss)
            is_best = not best_loss or validation_loss < best_loss
            if is_best:
                best_loss = validation_loss

            self.save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": validation_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, is_best)

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.manual_seed)
    t = ModelTrainer(args)
    
    if args.validation:
        t.test()
    else:
        t.train()
