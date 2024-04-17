import cv2
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from utils.utils import readImage, load_network
from models.CNN import ED_Model1
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

testList = [
    "datasets/fer2013/test/angry/PrivateTest_1054527.jpg",
    "datasets/fer2013/test/happy/PrivateTest_95094.jpg",
    "datasets/fer2013/test/disgust/PrivateTest_807646.jpg",
    "datasets/fer2013/test/fear/PrivateTest_518212.jpg",
    "datasets/fer2013/test/neutral/PrivateTest_59059.jpg",
    "datasets/fer2013/test/sad/PrivateTest_528072.jpg",
    "datasets/fer2013/test/surprise/PrivateTest_139065.jpg"
]

testList1 = ["datasets/maskedDataset/processedDataset/test/angry/PrivateTest_1054527_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/disgust/PrivateTest_807646_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/fear/PrivateTest_518212_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/happy/PrivateTest_95094_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/neutral/PrivateTest_59059_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/sad/PrivateTest_528072_N95.jpg",
            "datasets/maskedDataset/processedDataset/test/surprise/PrivateTest_139065_N95.jpg"]

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
dicton = {
            "angry": torch.tensor([1,0,0,0,0,0,0]),
            "disgust": torch.tensor([0,1,0,0,0,0,0]),
            "fear":torch.tensor([0,0,1,0,0,0,0]),
            "happy": torch.tensor([0,0,0,1,0,0,0]),
            "neutral":torch.tensor([0,0,0,0,1,0,0]),
            "sad":torch.tensor([0,0,0,0,0,1,0]),
            "surprise": torch.tensor([0,0,0,0,0,0,1])
            }

modelk = ED_Model1(in_channels=1, out_channels=7)
loadedModel = load_network(modelk, "saves/E_FocalFer2013Masked.pth")
target_layers = [modelk.features[-4]]
cam = GradCAMPlusPlus(model=modelk, target_layers=target_layers)
for idx, ii in enumerate(testList1):
    original, label = readImage(ii)
    feature = original.unsqueeze(0).requires_grad_()
    grayscale_cam = cam(input_tensor=feature, targets=None)
    heatmap_scaled = (grayscale_cam * 255).astype(np.uint8).squeeze()
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    original_np = original.squeeze().numpy() * 255
    original_rgb = cv2.merge([original_np, original_np, original_np]).astype(np.uint8)
    overlay = cv2.addWeighted(original_rgb, 0.8, heatmap_colored, 0.3, 0)
    cv2.imwrite("images/Grad_cam/M_{}.jpg".format(classes[idx]), overlay)

classes = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
dicton = {
            "angry": torch.tensor([1,0,0,0,0,0,0]),
            "disgust": torch.tensor([0,1,0,0,0,0,0]),
            "fear":torch.tensor([0,0,1,0,0,0,0]),
            "happy": torch.tensor([0,0,0,1,0,0,0]),
            "neutral":torch.tensor([0,0,0,0,1,0,0]),
            "sad":torch.tensor([0,0,0,0,0,1,0]),
            "surprise": torch.tensor([0,0,0,0,0,0,1])
            }

modelk = ED_Model1(in_channels=1, out_channels=7)
loadedModel = load_network(modelk, "saves/E_FocalFer2013Masked.pth")
target_layers = [modelk.features[-4]]
cam = GradCAMPlusPlus(model=modelk, target_layers=target_layers)
for idx, ii in enumerate(testList1):
    original, label = readImage(ii)
    feature = original.unsqueeze(0).requires_grad_()
    grayscale_cam = cam(input_tensor=feature, targets=None)
    heatmap_scaled = (grayscale_cam * 255).astype(np.uint8).squeeze()
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    original_np = original.squeeze().numpy() * 255
    original_rgb = cv2.merge([original_np, original_np, original_np]).astype(np.uint8)
    overlay = cv2.addWeighted(original_rgb, 0.8, heatmap_colored, 0.3, 0)
    cv2.imwrite("images/Grad_cam/M_{}.jpg".format(classes[idx]), overlay)
