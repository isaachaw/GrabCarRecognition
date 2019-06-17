import argparse

import os
import sys
from typing import List, Tuple, NoReturn

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import numpy as np

import random

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from resnet152 import resnet152
from load_data import CarData

from darknet import Darknet
from utils import non_max_suppression, rescale_boxes, pad_to_square

IMAGE_SIZE = 224
NUM_CLASSES = 196

MEAN_RGB = (0.485, 0.456, 0.406)
STDDEV_RGB = (0.229, 0.224, 0.225)

def parse_args():
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("--mode", type=str, help="one of the following modes: predict, predict_and_detect")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--image_path", type=str, help="path of the input image")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="the model directory (default: model)")
    parser.add_argument("--best_filename", type=str, default="best.tar",
                        help="filename of the best checkpoint (default: best.tar)")
    args = parser.parse_args()
    return args

class Predictor():
    
    def __init__(self, args):
        super(Predictor, self).__init__()
        
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Car Class labels

        self.car_data = CarData()

        # Prediction model

        self.model_predict = self.load_prediction()
        self.model_predict.eval()

        self._predict_image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
        ])

        # Detection model

        self.model_detect = self.load_detection()
        self.model_detect.eval()

        self._detection_image_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: pad_to_square(x)[0],
            lambda x: F.interpolate(x.unsqueeze(0), size=self.args.image_size, mode="nearest").squeeze(0)
        ])

    def load_prediction(self) -> Tuple[nn.Module, int, float]:
        """Load prediction model"""

        best_filename = os.path.join(self.args.model_dir, self.args.best_filename)

        if not os.path.exists(best_filename):
            raise Exception("Model doesn't exist.")

        state = torch.load(best_filename)
        print("Reloading model at epoch {}"
            ", with test error {}".format(
                state["epoch"],
                state["loss"]))
        model = resnet152(NUM_CLASSES).to(self.device)
        model.load_state_dict(state["state_dict"])
        best_epoch = state["epoch"]
        best_loss = state["loss"]

        return model

    def load_detection(self) -> nn.Module:
        """Load YOLO v3 model"""

        model = Darknet(config_path=self.args.model_def, img_size=self.args.image_size).to(self.device)
        if self.args.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.args.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.args.weights_path))

        return model

    def _load_image(self, image_path: str):
        image = Image.open(image_path)
        return image

    def _detect(self, image) -> List[Tuple[int]]:

        ori_image_height, ori_image_width = np.array(image).shape[:2]

        image = self._detection_image_transform(image)
        image = image.to(self.device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model_detect(image)
            outputs = non_max_suppression(outputs, conf_thres=self.args.conf_thres, nms_thres=self.args.nms_thres)

        bounding_boxes = []

        for detections in outputs:
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.args.image_size, (ori_image_height, ori_image_width))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    if cls_pred.item() in (2, 7): # 2 == car, 7 == truck
                        bounding_box = [x1, y1, x2, y2]
                        bounding_box = tuple(int(v.item()) for v in bounding_box)
                        bounding_boxes.append(bounding_box)

        return bounding_boxes

    def detect_and_predict(self, image):
        image = self._load_image(image_path)        
        bounding_boxes = self._detect(image)

        if len(bounding_boxes) == 0:
            print("No car detected")
            return

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        bbox_colors = random.sample(colors, len(bounding_boxes))

        plt.figure()
        ax = plt.subplot(111)
        ax.imshow(image)

        for idx, bounding_box in enumerate(bounding_boxes):
            subimage = image.crop(bounding_box)
            top5 = self._predict(subimage)

            car_class = self.car_data.class_names[top5.indices[0]]
            confidence = top5.values[0]

            # Create a Rectangle patch
            x1, y1, x2, y2 = bounding_box
            box_w = x2 - x1
            box_h = y2 - y1
            color = bbox_colors[idx]
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=car_class,
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.show()

    def _predict(self, image):

        image = self._predict_image_transform(image)
        image = image.to(self.device)
        image = image.unsqueeze(0)

        with torch.no_grad():
            logits = self.model_predict(image).squeeze()
            softmax = F.softmax(logits)
            top5 = softmax.topk(5)

        self.parse_topk_result(top5)
        return top5

    def predict(self, image_path: str):
        image = self._load_image(image_path)
        return self._predict(image)

    def parse_topk_result(self, topk) -> NoReturn:
        class_scores = list(zip(topk.values, topk.indices))

        print("Top {} results:".format(len(class_scores)))
        for idx, (k_score, k_idx) in enumerate(class_scores):
            print("[{}] Confidence: {:.4f}, Class: {} - {}".format(
                idx + 1, k_score.item(),
                k_idx.item(), self.car_data.class_names[k_idx.item()]))

if __name__ == "__main__":
    args = parse_args()

    image_path = args.image_path
    if not os.path.exists(image_path):
        raise Exception("File does not exist: {}".format(image_path))

    mode = args.mode

    if mode == "predict":
        t = Predictor(args)
        t.predict(image_path)
    elif mode == "predict_and_detect":
        t = Predictor(args)
        t.detect_and_predict(image_path)
