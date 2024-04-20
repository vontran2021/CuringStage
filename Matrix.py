import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable


from model import efficientnetv2_m as create_model


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = torch.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[t, p] += 1

    def summary(self):
        sum_TP = self.matrix.diag().sum().item()
        acc = 100 * sum_TP / self.matrix.sum().item()
        print("the model accuracy is ", acc)

        confusion_matrix = self.matrix / self.matrix.sum(1, keepdim=True) * 100
        confusion_matrix = torch.round(confusion_matrix * 100) / 100

        table = PrettyTable()
        table.field_names = [""] + [f"Stage {i}" for i in range(self.num_classes)]
        for i in range(self.num_classes):
            row = [f"Stage {i}"]
            for j in range(self.num_classes):
                row.append("{:.2f}".format(confusion_matrix[i, j].item()))
            table.add_row(row)
        print(table)

        table2 = PrettyTable()
        table2.field_names = ["Class", "Precision", "Recall", "F1-score"]
        for i in range(self.num_classes):
            TP = confusion_matrix[i, i]
            FP = confusion_matrix[i, :].sum() - TP  # Fix FP calculation
            FN = confusion_matrix[:, i].sum() - TP  # Fix FN calculation
            TN = confusion_matrix.sum() - (TP + FP + FN)

            Precision = TP / (TP + FP) if TP + FP != 0 else 0.
            Recall = TP / (TP + FN) if TP + FN != 0 else 0.
            F1_score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall != 0 else 0.

            table2.add_row(
                [f"Stage {i}", "{:.3f}".format(Precision), "{:.3f}".format(Recall), "{:.3f}".format(F1_score)])
        print(table2)


def weighted_average(model_weight_paths, device, validate_loader):
    total_confusion = ConfusionMatrix(num_classes=11, labels=[])

    for i, model_weight_path in enumerate(model_weight_paths):
        assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)

        model = create_model(num_classes=11)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.to(device)
        model.eval()

        json_label_path = '/test/TobaccoStage/Network/MobileNetV3/TMBclass_Flue_TobaccoStage.json'
        assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
        json_file = open(json_label_path, 'r')
        class_indict = json.load(json_file)
        labels = [label for _, label in class_indict.items()]
        confusion = ConfusionMatrix(num_classes=11, labels=labels)

        with torch.no_grad():
            for val_data in tqdm(validate_loader):
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predictions = torch.argmax(outputs, dim=1)
                confusion.update(predictions.to("cpu").numpy(), val_labels.to("cpu").numpy())

        print(f"\n--- Weighted Average {i + 1} ---")
        confusion.summary()

        # Accumulate confusion matrix
        for i in range(11):
            for j in range(11):
                total_confusion.matrix[i, j] += confusion.matrix[i, j]

    # Compute average confusion matrix
    total_confusion.matrix /= len(model_weight_paths)

    print("\n--- Overall Average ---")
    total_confusion.summary()


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    img_size = {"s": [224, 224], "m": [224, 224], "l": [384, 480]}
    num_model = "m"

    data_transform = transforms.Compose([transforms.Resize(img_size[num_model][1]),
                                         transforms.CenterCrop(img_size[num_model][1]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    image_path = r"/test/TobaccoStage/data_set/BB"
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform)

    batch_size = 64
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=8)

    model_weight_paths = ["/test/TobaccoStage/Network/EfficientNetV2/New_weights/M/TMB/checkpoint_TMB_200.pth",
                          "/test/TobaccoStage/Network/EfficientNetV2/New_weights/M/TMB/checkpoint_TMB_199.pth",
                          "/test/TobaccoStage/Network/EfficientNetV2/New_weights/M/TMB/checkpoint_TMB_198.pth",
                         "/test/TobaccoStage/Network/EfficientNetV2/New_weights/M/TMB/checkpoint_TMB_197.pth",
                         "/test/TobaccoStage/Network/EfficientNetV2/New_weights/M/TMB/checkpoint_TMB_196.pth"]

    weighted_average(model_weight_paths, device, validate_loader)
