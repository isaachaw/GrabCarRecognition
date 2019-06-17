from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

PREPROCESSED_TRAIN_DATA_PATH = "./data/preprocessed/train"
PREPROCESSED_VALIDATION_DATA_PATH = "./data/preprocessed/validation"

MEAN_RGB = (0.485, 0.456, 0.406)
STDDEV_RGB = (0.229, 0.224, 0.225)

def get_car_train_dataset(size) -> Dataset:
    _train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
    ])

    car_train_dataset = torchvision.datasets.ImageFolder(
        root=PREPROCESSED_TRAIN_DATA_PATH,
        transform=_train_transform)

    return car_train_dataset

def get_car_validation_dataset(size) -> Dataset:
    _validation_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
    ])

    car_train_dataset = torchvision.datasets.ImageFolder(
        root=PREPROCESSED_VALIDATION_DATA_PATH,
        transform=_validation_transform)

    return car_train_dataset
