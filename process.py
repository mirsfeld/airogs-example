from typing import Dict

import SimpleITK
import tqdm
import json
from pathlib import Path
import tifffile
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as functional
import classifier_models
from PIL import Image

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader
import cv2
import os

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import scale_boxes, non_max_suppression
from yolov5.utils.plots import save_one_box

def hist_equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    return img

def get_model(model_path):
    pretrained = ('pretrained' in model_path)
    if 'maxvit' in model_path:
        return classifier_models.MaxViT(input_channels=3, pretrained=pretrained), 1.25
    elif 'resnet' in model_path:
        return classifier_models.ResNet(pretrained=pretrained), 0.75
    elif 'efficientnet' in model_path:
        return classifier_models.EfficientNet(pretrained=pretrained), 1.25

def run_detection(image, device):
    original_image = image.copy()
    image = letterbox(image, new_shape=(640,640))[0].transpose((2,0,1))[::-1]
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).float()
    image /=255
    if len(image.shape)==3:
        image = image[None]
    model = DetectMultiBackend(weights='best.pt', device=torch.device(device), dnn=False, data=None, fp16=False)
    pred_0 = model(image, augment=False, visualize=False)
    pred = non_max_suppression(pred_0, conf_thres=0.1, max_det=1)
    det = pred[0]
    det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], original_image.shape).round()
    if len(det):
        *xyxy, conf, cls = reversed(det)[0]
        # print('Confidence yolov5:', conf.item())
        crop = save_one_box(xyxy, original_image, file='', BGR=False, pad=150, square=True, save=False)
        return {'crop': crop, 'conf': conf.item(), 'box_detected':True}
    else:
        pred = non_max_suppression(pred_0, conf_thres=0.01, max_det=1)
        det = pred[0]
        det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], original_image.shape).round()
        *xyxy, conf, cls = reversed(det)[0]
        # print('Confidence yolov5:', conf.item())
        return {'crop': original_image, 'conf':conf.item(), 'box_detected':False}



    # detected, conf, crop = detect_optic_disc(
    #     weights='best.pt',
    #     source = image,
    #     imgsz=(640,640),
    #     conf_thres=0.4,
    #     max_det=1,
    #     save_crop=False,
    #     save_txt=False,
    #     project='',
    #     line_thickness=0,
    #     hide_conf=True,
    #     hide_labels=True,
    #     name='',
    #     exist_ok=True,
    #     device=device
    # )
    # if detected:
    #     return image, conf
    # else:
    #     crop = hist_equalization(crop)
    #     return crop, conf

def crop_parameters(img_arr):
    indices_0 = np.where(np.any(img_arr>10, axis=(1,2)))[0]
    indices_1 = np.where(np.any(img_arr>10, axis=(0,2)))[0]
    
    h, w = indices_0[-1]-indices_0[0], indices_1[-1]-indices_1[0]
    # tuple of shape top, left, height, width
    if h > w:
        padding = int(np.ceil((h-w)/2))
        return indices_0[0], indices_1[0]-padding, h
    else:
        padding = int(np.ceil((w-h)/2))
        return indices_0[0]-padding, indices_1[0], w

def clahe(rgb_img):
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
    lab_img[...,0] = clahe.apply(lab_img[...,0])
    return cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

class DummyLoader(ImageLoader):
    @staticmethod
    def load_image(fname):
        return str(fname)


    @staticmethod
    def hash_image(image):
        return hash(image)


class airogs_algorithm(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        self._file_loaders = dict(input_image=DummyLoader())

        self.output_keys = ["multiple-referable-glaucoma-likelihoods", 
                            "multiple-referable-glaucoma-binary",
                            "multiple-ungradability-scores",
                            "multiple-ungradability-binary"]
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_full = classifier_models.MaxViT(pretrained=False)
        self.model_full.load_state_dict(torch.load('./weights_full_image.pth', map_location=torch.device('cpu')))
        self.model_full = self.model_full.to(self.device)

        self.model_cropped = classifier_models.MaxViT(pretrained=False)
        self.model_cropped.load_state_dict(torch.load('./weights_cropped_image.pth', map_location=torch.device('cpu')))
        self.model_cropped = self.model_cropped.to(self.device)

        model_paths = [
            # './log_data_augm/swintransformer_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold0.pth',
            # './log_data_augm/swintransformer_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold1.pth',
            # './log_data_augm/swintransformer_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold2.pth',
            # './log_data_augm/swintransformer_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold3.pth',
            # './log_data_augm/maxvit_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold0.pth',
            # './log_data_augm/maxvit_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold1.pth',
            # './log_data_augm/maxvit_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold2.pth',
            # './log_data_augm/maxvit_clahe_20epochs_lr5e-05_adamw_constant/best_model_fold3.pth',
            './weights/maxvit_clahe_20epochs_lr1e-05_adamw_linear_pretrained/best_model_fold0.pth',
            './weights/maxvit_clahe_20epochs_lr1e-05_adamw_linear_pretrained/best_model_fold1.pth',
            './weights/maxvit_clahe_20epochs_lr1e-05_adamw_linear_pretrained/best_model_fold2.pth',
            './weights/maxvit_clahe_20epochs_lr1e-05_adamw_linear_pretrained/best_model_fold3.pth',
            './weights/maxvit_clahe_20epochs_lr1e-05_adamw_linear_pretrained/best_model_fold4.pth',
            './weights/resnet50_clahe_20epochs_lr1e-05_adamw_exponential/best_model_fold0.pth',
            './weights/resnet50_clahe_20epochs_lr1e-05_adamw_exponential/best_model_fold1.pth',
            './weights/resnet50_clahe_20epochs_lr1e-05_adamw_exponential/best_model_fold2.pth',
            './weights/resnet50_clahe_20epochs_lr1e-05_adamw_exponential/best_model_fold3.pth',
            './weights/efficientnet_clahe_20epochs_lr0.0001_adamw_linear/best_model_fold0.pth',
            './weights/efficientnet_clahe_20epochs_lr0.0001_adamw_linear/best_model_fold1.pth',
            './weights/efficientnet_clahe_20epochs_lr0.0001_adamw_linear/best_model_fold2.pth',
            './weights/efficientnet_clahe_20epochs_lr0.0001_adamw_linear/best_model_fold3.pth',
            './weights/efficientnet_clahe_20epochs_lr0.0001_adamw_linear/best_model_fold4.pth'
        ]
        self.model_dict = {
        'model': [],
        'weight': []
        }
        for path in model_paths:
            model, weight  = get_model(path)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model_dict['model'].append(model.to(self.device))
            self.model_dict['weight'].append(weight)
    
    def load(self):
        for key, file_loader in self._file_loaders.items():
            fltr = (
                self._file_filters[key] if key in self._file_filters else None
            )
            self._cases[key] = self._load_cases(
                folder=Path("/input/images/color-fundus/"),
                file_loader=file_loader,
                file_filter=fltr,
            )

        pass
    
    def combine_dicts(self, dicts):
        out = {}
        for d in dicts:
            for k, v in d.items():
                if k not in out:
                    out[k] = []
                out[k].append(v)
        return out
    
    def process_case(self, *, idx, case):
        # Load and test the image(s) for this case
        if case.path.suffix == '.tiff':
            results = []
            with tifffile.TiffFile(case.path) as stack:
                for page in tqdm.tqdm(stack.pages):
                    input_image_array = page.asarray()
                    results.append(self.predict(input_image_array=input_image_array))
        else:
            input_image = SimpleITK.ReadImage(str(case.path))
            input_image_array = SimpleITK.GetArrayFromImage(input_image)
            results = [self.predict(input_image_array=input_image_array)]
        
        results = self.combine_dicts(results)

        # Test classification output
        if not isinstance(results, dict):
            raise ValueError("Expected a dictionary as output")

        return results

    def predict(self, *, input_image_array: np.ndarray) -> Dict:
        top, left, size = crop_parameters(input_image_array)
        # input_image_array = Image.fromarray(input_image_array)
        
        input_image_array = clahe(input_image_array)

        result_dict = run_detection(input_image_array, self.device)
        crop = Image.fromarray(result_dict['crop'])
        if not result_dict['box_detected']:
            classifier = self.model_full
            crop = functional.crop(crop, top, left, size, size)
        else:
            classifier = self.model_cropped
        crop = functional.resize(crop, size=(448,448))
        # input_image_array = Image.fromarray(input_image_array)
        # input_image_array = functional.to_tensor(input_image_array)

        # input_image_array = functional.crop(input_image_array, top, left, size, size)
        
        # input_image_array = functional.resize(input_image_array, size=(448,448))
        # input_image_array = torch.from_numpy(clahe(input_image_array.transpose(0,1).transpose(1,2).numpy())).reshape(1,3,224,224)
        crop = functional.to_tensor(crop)
        crop = crop.reshape(1,3,448,448)
        tmp = torch.stack([self.model_dict['weight'][i]*F.softmax(self.model_dict['model'][i](crop.to(self.device)), dim=1) for i in range(len(self.model_dict['model']))])
        # print(tmp.shape)
        pred = torch.mean(tmp, dim=0)
        # pred = classifier(crop.to(self.device))
        # pred = F.softmax(pred, dim=1)
        rg_likelihood = pred[:,1]
        rg_binary = (rg_likelihood>2/3)
        ungradability_score = 1-result_dict['conf']
        ungradability_binary = bool(ungradability_score>0.5)
        # From here, use the input_image to predict the output
        # We are using a not-so-smart algorithm to predict the output, you'll want to do your model inference here

        # Replace starting here
        # rg_likelihood = ((input_image_array - input_image_array.min()) / (input_image_array.max() - input_image_array.min())).mean()
        # rg_binary = bool(rg_likelihood > .2)
        # ungradability_score = rg_likelihood * 15
        # ungradability_binary = bool(rg_likelihood < .2)
        # to here with your inference algorithm

        out = {
            "multiple-referable-glaucoma-likelihoods": rg_likelihood.item(),
            "multiple-referable-glaucoma-binary": rg_binary.item(),
            "multiple-ungradability-scores": ungradability_score,
            "multiple-ungradability-binary": ungradability_binary
        }
        # print(out)

        return out

    def save(self):
        for key in self.output_keys:
            with open(f"/output/{key}.json", "w") as f:
                out = []
                for case_result in self._case_results:
                    out += case_result[key]
                json.dump(out, f)


if __name__ == "__main__":
    airogs_algorithm().process()
