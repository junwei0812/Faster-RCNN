from PIL import Image
from torch.utils.data import Dataset
import torchvision
import torch
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from torchvision import transforms
class VOCDataset(Dataset):
    CLASS_NAME = (
        "__background__",
        "pottedplant",
        "person",
        "horse",
        "chair",
        "car"
    )
    def __init__(self, root_dir, resize = [1080,720], split='trainval', use_difficult=False, transforms = None):
        super(VOCDataset,self).__init__()
        self.root = root_dir
        self.split = split
        self.use_difficult = use_difficult

        self._ann_path = os.path.join(self.root, 'Annotations', "%s.xml")
        self._img_path = os.path.join(self.root, 'JPEGImages', "%s.jpg")
        self._imgset_path = os.path.join(self.root,'ImageSets', 'Main', "%s.txt")

        with open(self._imgset_path % self.split) as f:
            self.img_ids = f.readlines()
            self.img_ids = [x.strip() for x in self.img_ids]
            self.name2id = dict(zip(VOCDataset.CLASS_NAME, range(len(VOCDataset.CLASS_NAME))))
            self.new_size = tuple(resize)
            self.mean = [0.485,0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            print("INFO: VOC Dataset init finished !")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img,self.new_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = self._read_img_rgb(self._img_path % img_id)

        anno = ET.parse(self._ann_path % img_id).getroot()
        boxes = []
        classes = []

        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1  # 由于像素是网格存储，坐标2实质表示第一个像素格，所以-1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name = obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])  # 将类别映射回去

        boxes = np.array(boxes, dtype=np.float32)

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.tensor(classes)

        return img, boxes, classes

if __name__ == '__main__':
    dataset = VOCDataset(r'D:\VOCdevkit\VOC2007',split='train')
    img, boxes, classes = dataset[0]
    print(img.shape)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=10,num_workers=2)
    print(train_loader)

