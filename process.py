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
import models
from PIL import Image

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from evalutils.io import ImageLoader
import cv2

def crop_parameters(img_arr):
    indices_0 = np.where(np.any(img_arr!=0, axis=(1,2)))[0]
    indices_1 = np.where(np.any(img_arr!=0, axis=(0,2)))[0]
    
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
        
        self.model = models.MaxViT(pretrained=False)
        self.model.load_state_dict(torch.load('./best_model_fold0.pth', map_location=torch.device('cpu')))
        self.model = self.model.to(self.device)
    
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
        input_image_array = Image.fromarray(input_image_array)
        input_image_array = functional.to_tensor(input_image_array)

        input_image_array = functional.crop(input_image_array, top, left, size, size)
        
        input_image_array = functional.resize(input_image_array, size=(448,448))
        # input_image_array = torch.from_numpy(clahe(input_image_array.transpose(0,1).transpose(1,2).numpy())).reshape(1,3,224,224)
        input_image_array = input_image_array.reshape(1,3,448,448)
        pred = self.model(input_image_array.to(self.device))
        pred = F.softmax(pred, dim=1)
        rg_likelihood = pred[:,1]
        rg_binary = torch.argmax(pred[0]).bool()
        ungradability_score = 0
        ungradability_binary = False
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
