# Grad-CAM--Understand-deep-learning-from-higher-view
Gradient Class Activation Map: Visualize the model's prediction and can help you understand neural network models better

Thanks to Selvaraju's research: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

This repository realizes Grad-CAM with Pytorch on Pytorch based models (pretrained models are from timm). To test the visualized results, I use resenet34 and resnet10 models.

Required Library: torch, matplotlib, torchvision, PIL, numpy

Example (from restnet10) (The color from blue -> green -> red represents the focus of model on the picture low -> median -> high):
![Grad-CAM-with-resnet10](graphs/resnet10-targe_layer-2.png)

