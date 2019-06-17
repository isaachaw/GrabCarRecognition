import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
from scipy.io import loadmat

DEV_KIT_PATH = "./data/devkit"
TRAIN_DATA_PATH = "./data/cars_train"
PREPROCESSED_TRAIN_DATA_PATH = "./data/preprocessed/train"
PREPROCESSED_VALIDATION_DATA_PATH = "./data/preprocessed/validation"

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
CROP_MARGIN = 16

class CarData():
    def __init__(self,
                 input_shape=(INPUT_WIDTH, INPUT_HEIGHT),
                 crop_margin=CROP_MARGIN,
                 train_ratio=0.8,
                 seed=42):
        self.input_shape = input_shape
        self.crop_margin = crop_margin

        self._train_ratio = train_ratio
        self._random = np.random.RandomState(seed)

        self.train_annotations = self.load_annotations()
        self.class_names = self.load_class_names()

    def load_annotations(self):
        annotations_mat_path = os.path.join(DEV_KIT_PATH, "cars_train_annos.mat")
        annotations_mat = loadmat(annotations_mat_path)
        annotations = annotations_mat["annotations"][0]

        bbox_x1s = annotations["bbox_x1"].astype(np.uint32)
        bbox_y1s = annotations["bbox_y1"].astype(np.uint32)
        bbox_x2s = annotations["bbox_x2"].astype(np.uint32)
        bbox_y2s = annotations["bbox_y2"].astype(np.uint32)
        classes = annotations["class"].astype(np.uint32)
        fnames = [fname[0] for fname in annotations["fname"]]

        df = pd.DataFrame(data={
                "bbox_x1": bbox_x1s,
                "bbox_y1": bbox_y1s,
                "bbox_x2": bbox_x2s,
                "bbox_y2": bbox_y2s,
                "class": classes,
            },
            index=fnames)

        return df

    def load_class_names(self):
        meta_mat_path = os.path.join(DEV_KIT_PATH, "cars_meta.mat")
        meta_mat = loadmat(meta_mat_path)
        meta = meta_mat["class_names"][0]
        class_names = [class_name[0] for class_name in meta]

        return class_names

    def load_image(self, image_path, bbox=None, resize=None):
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]

        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        crop_x1 = max(0, bbox_x1 - self.crop_margin)
        crop_y1 = max(0, bbox_y1 - self.crop_margin)
        crop_x2 = min(bbox_x2 + self.crop_margin, image_width)
        crop_y2 = min(bbox_y2 + self.crop_margin, image_height)
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        input_width, input_height = self.input_shape
        resized_image = cv2.resize(cropped_image, (input_height, input_width))

        return resized_image
        # return cropped_image

    def prepare_train_data(self):
        print(f"Preparing train/validation data...")

        grouped = self.train_annotations.groupby("class", as_index=False)
        classes = list(grouped.groups.keys())

        for class_ in classes:
            for row in grouped.get_group(class_).itertuples():
                fname = row[0]
                bbox = (row[1], row[2], row[3], row[4])
                class_ = row[5]

                image_path = os.path.join(TRAIN_DATA_PATH, fname)
                image = self.load_image(image_path, bbox)

                # split into train/validation
                if self._random.rand() < self._train_ratio:
                    path = PREPROCESSED_TRAIN_DATA_PATH
                else:
                    path = PREPROCESSED_VALIDATION_DATA_PATH

                save_dir = os.path.join(path, f"{class_:05d}")
                pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

                save_path = os.path.join(save_dir, fname)
                cv2.imwrite(save_path, image)

    def show_sample(self, i=None):
        if i is None:
            i = np.random.randint(len(self.train_annotations))

        row = self.train_annotations.iloc[i]
        image_name = row.name
        x1 = row["bbox_x1"]
        y1 = row["bbox_y1"]
        x2 = row["bbox_x2"]
        y2 = row["bbox_y2"]
        class_ = row["class"]

        image_path = os.path.join(TRAIN_DATA_PATH, image_name)
        image = cv2.imread(image_path)

        # draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        plt.imshow(image)
        plt.title(self.class_names[class_ - 1])
        plt.show()
        plt.close()

if __name__ == "__main__":
    d = CarData()
    d.prepare_train_data()
