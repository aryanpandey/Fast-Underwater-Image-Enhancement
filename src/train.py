import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import EUVP
from model import Shallow_UWNet
import argparse
from torchvision import models
from torch import optim
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGG_loss(nn.Module):
    def __init__(self, model, config):
        super(VGG_loss, self).__init__()
        self.features = nn.Sequential(*list(model.children())[0][:-3]).to(config)
    def forward(self, x):
        return self.features(x)

class combinedloss(nn.Module):
    def __init__(self, config):
        super(combinedloss, self).__init__()
        vgg = models.vgg19_bn(pretrained=True)
        self.vggloss = VGG_loss(vgg, config)
        for param in self.vggloss.parameters():
            param.requires_grad = False
        self.mseloss = nn.MSELoss().to(config)
        self.l1loss = nn.L1Loss().to(config)

    def forward(self, out, label):
        inp_vgg = self.vggloss(out)
        label_vgg = self.vggloss(label)
        mse_loss = self.mseloss(out, label)            # Pixel-Wise Loss
        vgg_loss = self.l1loss(inp_vgg, label_vgg)     # VGG Perpetual Loss
        total_loss = mse_loss + vgg_loss
        return total_loss, mse_loss, vgg_loss

def main(args):
    now = datetime.now()
    date = now.strftime("%d-%m-%Y %H:%M:%S").split(' ')[0]
    model_save_path = 'Training_Experiments/' + date + args.comments + '.pth'
    dataset = EUVP(root_dir=args.root_dir, img_dim=[args.img_width, args.img_height])
    val_size = int(args.val_size*len(dataset))
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])
    
    trainloader = DataLoader(trainset, args.batch_size, shuffle = True, num_workers = 4)
    valloader = DataLoader(valset, 2, shuffle = False, num_workers = 4)

    model = Shallow_UWNet(args.initial_conv_filters, args.mid_conv_filters, args.network_depth).to(device)
    criterion = combinedloss(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    best_val_loss = 999999
    for i in range(args.epochs):
        running_loss = 0
        model.train()
        for step, (blurred, high_res) in enumerate(trainloader):
            blurred = blurred.to(device)
            high_res = high_res.to(device)

            output = model(blurred)
            loss, mse_l, vgg_l = criterion(output, high_res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if (step+1)%500 == 0:
                print("Epoch: {}/{}, Step: {}/{}, Combined Loss: {:.3f}".format(
                    i+1, args.epochs, step+1, len(trainloader), running_loss/(step+1)))

        model.eval()
        val_loss = 0
        for step, (blurred, high_res) in enumerate(valloader):
            blurred = blurred.to(device)
            high_res = high_res.to(device)

            output = model(blurred)
            loss, mse_l, vgg_l = criterion(output, high_res)
            val_loss += loss.item()

        print("Epoch: {}, Validation Loss: {}".format(i+1, val_loss/len(valloader)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default = 'Data/EUVP_Dataset/Paired', type = str)
    ap.add_argument('--img_height', default = 256, type = int)
    ap.add_argument('--img_width', default = 256, type = int)
    ap.add_argument('--batch_size', default = 8, type=int)
    ap.add_argument('--lr', default = 2e-4, type = int)
    ap.add_argument('--epochs', default=50, type = int)
    ap.add_argument('--val_size', default = 0.15, type=int)
    ap.add_argument('--initial_conv_filters', default = 64, type = int)
    ap.add_argument('--mid_conv_filters', default = 64, type = int)
    ap.add_argument('--network_depth', default = 2, type = int)
    ap.add_argument('--comments', default='', type = str)
    
    print("Training Device: ", device)
    args = ap.parse_args()
    main(args)
