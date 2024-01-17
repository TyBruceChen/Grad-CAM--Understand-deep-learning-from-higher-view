from torch import nn
import torch
from timm import create_model
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
  def __init__(self,model,
               img_path:str,
               layer_idx: int = 2,
               input_shape = (224,224)
               ):
    """
    params:
    layer_idx: the index of the layer where you want to visulize (it count
      from classifier to input layer).
    input_shape: the image shape to put into the model
    model: the model you want to visualize
    img_path: the path of the tested image
    """
    self.model = model
    self.layer_idx = -layer_idx
    self.img_path = img_path
    self.input_shape = input_shape
  def __call__(self):
    model = self.model

    extractor = nn.Sequential(*list(model.children())[:self.layer_idx]) #truncate the model from where your specified idx
    classifier = nn.Sequential(*list(model.children())[self.layer_idx:])
    #print(extractor)

    img = Image.open(img_path).resize(self.input_shape)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    img = torch.unsqueeze(img, 0) #preprocess the image to tensor: (1,C,H,W)
    img.requires_grad = True

    result = model(img)
    result = result.argmax()  #get the prediction category
    class_Idx = result  #the grad-cam will visualize the prediction towards a specific category. (Here is the model's prediction)

    activations = extractor(img)  #get the activations (output features) from target layer
    activations.retain_grad() #pytorch will automatically free the parameters' gradients that are not provided by user
                                #so here it needs to be specified to keep the gradient of activation w.r.t prediction logits

    print(f'Activation Shape:{activations.shape}')

    prediction_logits = classifier(activations) #the activation is fed into rest layers to get the prediction tensor

    prediction_logits = prediction_logits[:,class_Idx]  #only use the specific class to back propagate

    grad_output = torch.ones_like(prediction_logits)
    print(grad_output.shape)
    prediction_logits.backward(gradient = grad_output)  #according to pytorch, backward() should specify
                                                          #with a tensor which its length is the same as backward tensor
                                                            #when the tensor contains more than one number
    d_act = activations.grad  #get the gradient of activation from target layer w.r.t. the specified category

    d_act = d_act.permute(0,2,3,1)  #(1,C,H,W) -> (1,H,W,C)
    activations = activations.permute(0,2,3,1)
    pooled_grads = torch.mean(d_act,dim = (0,1,2))  #according to the paper, the pooling happens all axis except the channel dim

    heatmap = activations.detach().numpy()[0] #for tensors where its requires_grad = True, need detach() function to convert to ndarray
    pooled_grads = pooled_grads.numpy()

    #back propagate
    for  i in range(d_act.shape[-1]):
      heatmap[:,:,i] *= pooled_grads[i]
    heatmap = np.mean(heatmap, axis = -1)
    self.heatmap = np.uint8(255*np.maximum(heatmap,0)/np.max(heatmap))
  def origin_cam_visualization(self):
    plt.matshow(self.heatmap)
    plt.show
  def imposing_visualization(self):
    alpha = 0.6 #how much CAM will overlap on original image
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:,:3]
    jet_colors = (jet_colors*256).astype(np.uint8)  #generate a color (RGB) image which has small H and W
                                                      # and maps the intensity to color from red to blue

    jet_heatmap = (jet_colors[self.heatmap] * alpha).astype(np.uint8)
    #print(jet_heatmap.shape)
    jet_heatmap = Image.fromarray(jet_heatmap).resize(self.input_shape)
    img = Image.open(self.img_path).resize(self.input_shape)

    jet_heatmap = np.asarray(jet_heatmap)
    img_cam = np.asarray(img) + np.asarray(jet_heatmap)
    plt.subplot(1,2,1)
    plt.imshow(img_cam/255) #print the overlapped image (origin + cam)
    plt.subplot(1,2,2)
    plt.imshow(jet_heatmap/255) #print the cam
