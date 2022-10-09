# Fast-Underwater-Image-Enhancement
A Project to perform fast inference for Underwater Image Enhancement

Reference Paper: [Shallow-UWNet](https://arxiv.org/abs/2101.02073)

Please install torch separately using the given command. Make sure you have Cuda 10.1 on your system for the model to use the GPU on the system, else it will use the CPU

Command: pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

The inference.py script in the src folder will load the trained model onto the system and perform inference on the test set.

Please make sure that the 'data' folder given in the [drive](https://drive.google.com/file/d/1l_AooRLa2h2Uz7qJQ2o51BNAWIcZ9mXP/view?usp=sharing) is in the same directory as the inference.py script

If you want to train the model, please ensure you have a 16GB GPU to train the model on and run train.py. All the arguments of train.py have default values so simply running "python train.py" will work. Following are the arguments that train.py can take in:

--root_dir : the directory where the paired dataset will be available in the EUVP format

--img_height : Image Height after resizing

--img_width : Image Width after resizing

--batch_size: Batch size while training

--lr : Learning Rate of the model

--epochs: Epochs while training

--val_size: fraction of dataset used for validation

--initial_conv_filters: Number of filters in the initial convolution layer of the conv block

--mid_conv_filters: Number of filters in the middle convolution layer of the conv block

--network_depth: Total Number of ConvBlocks - 1 (So 3 ConvBlocks means a network depth of 2)

--comments: Any comments that you want to make for the training iteration. The comments get appended to the file name of the weights

The inference scipt inference.py takes in the following arguments:

--root_dir : the directory where the paired dataset will be available in the EUVP format

--img_height : Image Height after resizing

--img_width : Image Width after resizing

--initial_conv_filters: Number of filters in the initial convolution layer of the conv block

--mid_conv_filters: Number of filters in the middle convolution layer of the conv block

--network_depth: Total Number of ConvBlocks - 1 (So 3 ConvBlocks means a network depth of 2)

--weights_path: Path to the weights file that can be loaded to the model

