import torch
import time
import torch
from tqdm import tqdm

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
# Models and functions
from .timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from .timm.data import Dataset, DatasetTar, create_loader, resolve_data_config
from .timm.optim import create_optimizer

# Batch norm fusion
from .pytorch_bn_fusion.bn_fusion import fuse_bn_recursively

# Distiller Quantization

# import distiller as ds
# from distiller.distiller.data_loggers import collector_context
# from distiller.distiller import file_config

#from mobilenetv2.models.imagenet import mobilenetv2

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model', '-m', metavar='MODEL', default='efficientnet_b0',
                    help='model architecture (default: efficientnet_b0)')
parser.add_argument('--img-size', default=None, type=int, dest='img_size',
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--fuse-bn', default='', type=str, metavar='PATH', dest='fuse_bn',
                    help='Fuse BN to Conv layers')
parser.add_argument('--quant-aware',default='', type=str, metavar='PATH',
                    dest='quant_aware', help='Path to quant_aware_linear YAML config file')

args = parser.parse_args()
# create model
# model = create_model(
#     args.model,
#     num_classes=1000,
#     in_chans=3,
#     pretrained=False)

#model = mobilenetv2()
#model.load_state_dict(torch.load('mobilenetv2/pretrained/mobilenetv2_1.0-0c6065bc.pth'))


# # Load in pretrained model
# load_checkpoint(model, args.checkpoint, False)
model.to(device)


if args.fuse_bn:
    model = fuse_bn_recursively(model)

if args.quant_aware:
    print("-> Applying quant-aware...")
    optimizer = create_optimizer(args, model)
    compression_scheduler = file_config(model, optimizer, args.quant_aware, None)

# dummy_input = torch.randn(128, 3, args.img_size, args.img_size).to(device)

# traced_script_module=torch.jit.trace(model,dummy_input)
# traced_script_module.save(args.model + '.pt')

# torch_out = model(dummy_input)

# torch.onnx.export(model,
#                 dummy_input,
#                 args.model+".onnx",
#                 export_params=True,        # store the trained parameter weights inside the model file
#                 opset_version=10,          # the ONNX version to export the model to
#                 do_constant_folding=True,  # whether to execute constant folding for optimization
#                 input_names = ['input'],   # the model's input names
#                 output_names = ['output'] # the model's output names
#                 # dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                 #             'output' : {0 : 'batch_size'}}
#                 )


# cuDnn configurations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

#name = model.name
#print("     + {} Speed testing... ...".format(name))
model = model.to(device)
random_input = torch.randn(1,3,args.img_size, args.img_size).to(device)

model.eval()

time_list = []
for i in tqdm(range(10000)):
    torch.cuda.synchronize()
    tic = time.time()
    model(random_input)
    torch.cuda.synchronize()
    # the first iteration time cost much higher, so exclude the first iteration
    #print(time.time()-tic)
    time_list.append(time.time()-tic)
time_list = time_list[:]
print("     + Done 10000 iterations inference !")
print("     + Total time cost: {}s".format(sum(time_list)))
print("     + Average time cost: {}s".format(sum(time_list)/10000))
print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))
