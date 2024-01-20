# @author Maximus Mutschler and Nathan Inkawhich

from __future__ import print_function
import os

import matplotlib.pyplot as plt
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

print(os.getcwd())

# Configuration Parameters
image_to_attack_path = "data/image_to_attack.npy"
pretrained_model = "data/lenet_mnist_model.pth"
use_cuda = True
learning_rate = 0.01
max_opt_steps_per_class = 1000
reg_parameter = 0.01
# grad_clip = 0.05
reg_mode = "L1"

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# laod image to attack
np_image_to_attack, label_of_image_to_attack = numpy.load(image_to_attack_path, allow_pickle=True)
image_to_attack = torch.unsqueeze(torch.Tensor(np_image_to_attack), dim=0).to(device)

# plot image to attack
plt.imshow(np_image_to_attack.squeeze(), cmap="gray")
plt.title(f"True Class: {label_of_image_to_attack}")
plt.xticks([], [])
plt.yticks([], [])
plt.show()

target_classes = list(range(10))
target_classes.remove(label_of_image_to_attack)


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Initialize the network
model = Net().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout Layers
model.eval()


def reg_loss(data, image_to_attack, reg_parameter, mode):
    """
    :param image_as_params: copy of the image to attack on, which the optimizer uses as params
    :param image_to_attack: the original unchanged image
    :param reg_parameter:
    :param mode: either "None" for no regularization (return 0)
                or "L2" for L2 regularization
                or "L1" for L1 regularization
                or "L12" for L1 and L2 regularization with the same parameter
                :return: the regularization loss which will be later added to the network loss
    """
    if mode == "None":
        return 0
    elif mode == "L2":
        return reg_parameter * torch.sum(torch.pow(data - image_to_attack, 2))
    elif mode == "L1":
        return reg_parameter * torch.sum(torch.abs(data - image_to_attack))
    elif mode == "L12":
        return reg_parameter * torch.sum(torch.pow(data - image_to_attack, 2)) + reg_parameter * torch.sum(torch.abs(data - image_to_attack))

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# restores the tensors to their original scale
def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# List that holds the images of all succesfull attacks
attacked_images = []

for target_class in target_classes:
    ###
    #    TODO
    #    Use a copy of the image_to_attack as parameters.
    #    Then use SGD without momentum to change those until the correct class in predicted.
    #    add images that resemble a successful attack to attacked_images
    ###
    
    # copy image to attack
    image_as_params = torch.clone(image_to_attack).detach()
    image_as_params.to(device)
    image_as_params.requires_grad = True
    
    for step in range(max_opt_steps_per_class):
        # zero all gradients
        model.zero_grad()
        
        # forward pass
        output = model(image_as_params)
        
        # calculate loss
        loss = F.nll_loss(output, torch.tensor([target_class]).to(device))
        
        # calculate gradients
        loss.backward()
        
        # add regularization loss
        loss += reg_loss(image_as_params, image_to_attack, reg_parameter, reg_mode)
        
        # update image
        image_as_params.data = image_as_params.data - learning_rate * image_as_params.grad.data
        
        # clip image
        image_as_params.data = torch.clamp(image_as_params.data, 0, 1)
        
        # check if attack was successful
        if torch.argmax(output).item() == target_class:
            attacked_images.append(image_as_params)
            break



###
#    TODO
#    Todo plot all images in attacked_images with their corresponding classed in addition to the original image.
#    This should be on Figure
###
for img in attacked_images:
    # img = torch.clone(img).detach()
    # img.to(device)
    cpu_img = img.detach().cpu()
    plt.imshow(cpu_img.squeeze(), cmap="gray")
    plt.title(f"True Class: {label_of_image_to_attack}, Predicted Class: {torch.argmax(model(img)).item()}")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(f"src/plots/ClassSpecificAttack_{label_of_image_to_attack}_{torch.argmax(model(img)).item()}.png")
    plt.show()

