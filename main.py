from __future__ import print_function
import argparse
from math import log10
import os
import math
import torch
#import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import BBBNet
from utils.BBBlayers import GaussianVariationalInference
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=3, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=1024, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--num_samples', default=10, type=int, help='Number of samples')
parser.add_argument('--beta_type', default="Blundell", type=str, help='Beta type')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


global opt
opt = parser.parse_args()
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = BBBNet(upscale_factor=opt.upscale_factor).to(device)

vi = GaussianVariationalInference(torch.nn.MSELoss())

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

#load a model from checkpoint

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(opt.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))



def train(epoch):
    epoch_loss = 0
    m = math.ceil(len(train_set) / opt.batch_size)
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)



        if opt.beta_type is "Blundell":
            beta = 2 ** (m - (iteration + 1)) / (2 ** m - 1)
        elif opt.beta_type is "Soenderby":
            beta = min(epoch / (opt.num_epochs // 4), 1)
        elif opt.beta_type is "Standard":
            beta = 1 / m
        else:
            beta = 0

        outputs, kl = model.probforward(input)
        optimizer.zero_grad()
        loss = vi(outputs, target,kl,beta )
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    model.eval()
    m = math.ceil(len(train_set) / opt.batch_size)
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            if opt.beta_type is "Blundell":
                beta = 2 ** (m - (opt.testBatchSize + 1)) / (2 ** m - 1)
            elif opt.beta_type is "Soenderby":
                beta = min(epoch / (opt.num_epochs // 4), 1)
            elif opt.beta_type is "Standard":
                beta = 1 / m
            else:
                beta = 0

            prediction, kl = model.probforward(input)
            mse = vi(prediction, target, kl, beta)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_checkpoint(state):
    model_out_path = "./checkpoints/"+"model_epoch_{}.pth".format(epoch)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for epoch in range(1, opt.num_epochs + 1):
    train(epoch)
    test()
    #checkpoint(epoch)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': model,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    })

