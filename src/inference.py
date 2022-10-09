import torch
from model import Shallow_UWNet
from dataset import EUVP
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import argparse
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def img_denorm(img, mean, std):
    #for ImageNet the mean and std are:
    #mean = np.asarray([ 0.485, 0.456, 0.406 ])
    #std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    
    return(res)
    
def inference(args):
    testset = EUVP(root_dir=args.root_dir, img_dim=[args.img_width, args.img_height], train=False)
    testloader = DataLoader(testset, 1, shuffle = False, num_workers=4)
    
    model = Shallow_UWNet(args.initial_conv_filters, args.mid_conv_filters, args.network_depth).to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    
    results = []
    
    for i, (blurred, high_res) in enumerate(testloader):
        blurred = blurred.to(device)
        high_res = high_res.to(device)
        
        output = model(blurred)
        
        results.append(blurred.squeeze())
        results.append(output.squeeze())
        results.append(high_res.squeeze())
        
        if (i+1)%5 == 0:
            results = torch.stack(results, dim = 0)
            frame = make_grid(results.detach().cpu(), padding = 10, nrow=3, normalize=True, value_range=(-1, 1))
            save_image(frame, 'Test_Outputs/generated_'+str(i-3)+'_to_'+str(i+1)+'.jpg')

            results = []
        
        frame = img_denorm(output.squeeze(), mean = 0.5, std=0.5)
        save_image(frame, 'Individual_Outputs/'+ str(i+1) + '.jpg')
            
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root_dir', default = 'Data/EUVP_Dataset/test_samples', type = str)
    ap.add_argument('--img_height', default = 256, type = int)
    ap.add_argument('--img_width', default = 256, type = int)
    ap.add_argument('--initial_conv_filters', default = 64, type = int)
    ap.add_argument('--mid_conv_filters', default = 64, type = int)
    ap.add_argument('--network_depth', default = 2, type = int)
    ap.add_argument('--weights_path', default = 'Training_Experiments/03-10-2022.pth', type = str)
    ap.add_argument('--comments', default = '_img_256', type=str)
    
    args = ap.parse_args()
    inference(args)