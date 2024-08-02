import numpy as np
from skimage.transform import rescale
from torchvision.transforms import Compose
import torchvision
import numpy as np
import torch
import torch.nn.functional as F


from dataset import TomoDetectionDataset

"""
Augmentations during training phase:
    *   RandomScale()       ->  The dimensions of the input image are being multiplied 
                                with a random float number in [0.9 1.3].
    
    *   RandomCrop()        ->  The random scaled image is being cropped with a random way that 
                                contains though the ground truth bounding box of the corresponding image.
                                The cropped image has now these dimensions (1056, 672).
    
    *   RandomHorizFlip()   ->  The cropped image is being flipped horizontally with a probability of 50%.


Augmentations during validation and test phase:
    
    *   RandomCrop()        ->  The input image is being cropped with a random way that 
                                contains though the ground truth bounding box of the corresponding image.
                                The cropped image has now these dimensions (1056, 672).

"""

def transforms(train=True):
    if train:
        return Compose([
            # Normalize(mean=0.1271, std=0.1570),
            RandomScale(scale=0.1),
            RandomCrop((TomoDetectionDataset.img_height, TomoDetectionDataset.img_width)),
            RandomHorizFlip(prob=0.5),
            # transforms.ToTensor(),

            # NumpyToTensor(),
            # transforms.Normalize(mean=[0.4592], std=[0.22561]),
            # TensorToNumpy(),

            # Normalize(mean=0.1271, std=0.1570),
            ])
    else:
        return RandomCrop(
            (TomoDetectionDataset.img_height, TomoDetectionDataset.img_width),
            random=False,
        )


class RandomScale(object):

    def __init__(self, scale):
        assert isinstance(scale, (float, tuple))
        if isinstance(scale, float):
            assert 0.0 < scale < 1.0
            # self.scale = (1.0 - scale, 1.0 + scale)  # for scale = 0.2 then self.scale = [0.8, 1.2]
            self.scale = (1.0 - scale, 1.2 + scale)     # for scale = 0.1 then self.scale = [0.9 1.3]
        else:
            assert len(scale) == 2
            assert 0.0 < scale[0] < scale[1]
            self.scale = scale

    def __call__(self, sample):
        image, boxes = sample

        # # don't augment normal cases
        # if len(boxes["X"]) == 0:
        #     return image, boxes

        sample_scale = np.random.rand()
        # delete after:
        # print("sample_scale:",sample_scale)
        sample_scale = sample_scale * (self.scale[1] - self.scale[0]) + self.scale[0]

        # filter when down-sampling the image to avoid aliasing artifacts:
        if sample_scale < 1:
            apply_anti_aliasing = True
        else:
            apply_anti_aliasing = False

        scaled = rescale(
            image, sample_scale, multichannel=False, mode="constant", anti_aliasing=apply_anti_aliasing
        )

        boxes["X"] = [int(x * sample_scale) for x in boxes["X"]]
        boxes["Y"] = [int(y * sample_scale) for y in boxes["Y"]]
        boxes["Width"] = [int(w * sample_scale) for w in boxes["Width"]]
        boxes["Height"] = [int(h * sample_scale) for h in boxes["Height"]]

        return scaled, boxes


class RandomCrop(object):

    def __init__(self, crop_size, random=True):
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.random = random

    def __call__(self, sample):
        image, boxes = sample

        h = image.shape[0]
        w = image.shape[1]
        y_max = max(h - self.crop_size[0], 1)
        x_max = max(w - self.crop_size[1], 1) // 2
        if image[h // 2, self.crop_size[1]] == 0:
            x_max //= 2
        y_min = x_min = 0
        x_max_box = 0

        # don't crop boxes
        margin = 16
        if len(boxes["X"]) > 0:
            y_min_box = np.min(np.array(boxes["Y"]) - np.array(boxes["Height"]) // 2)
            x_min_box = np.min(np.array(boxes["X"]) - np.array(boxes["Width"]) // 2)
            y_max_box = np.max(np.array(boxes["Y"]) + np.array(boxes["Height"]) // 2)
            x_max_box = np.max(np.array(boxes["X"]) + np.array(boxes["Width"]) // 2)
            # min and max of coordinates for image cropping
            y_min = max(y_min, min(h, y_max_box + margin) - self.crop_size[0])
            x_min = max(x_min, min(w, x_max_box + margin) - self.crop_size[1])
            y_max = min(y_max, max(0, y_min_box - margin))
            x_max = min(x_max, max(0, x_min_box - margin))
            if x_max <= x_min:
                x_max = x_min + 1
            if y_max <= y_min:
                y_max = y_min + 1

        # this branch is for training set:
        if self.random:
            y_offset = np.random.randint(y_min, y_max)
            x_offset = np.random.randint(x_min, x_max)
            # delete after:
            # print("y_offset:",y_offset)
            # print("x_offset:",x_offset)

            # check if the current image (of training set) has a normal volume (and not a biopsied one):
            if len(boxes["X"]) == 0:

                crop_offset = (h // 2) - (self.crop_size[0] // 2)
                y_offset = np.random.randint(0, 2 * crop_offset )

                # set a x-axis margin for randomly cropping:
                # this is the maximum value, margin can not be grater than that:
                margin_x = 100

                # checking the laterality of the image: {"R","L"}:
                img_laterality = "R" if image[h // 2, 0] == 0 else "L"

                if img_laterality == "L":
                    x_offset = np.random.randint(0, margin_x)
                else:
                    # x_offset = np.random.randint(w - self.crop_size[1], w - self.crop_size[1] + margin_x)
                    x_offset = w - self.crop_size[1] - np.random.randint(0, margin_x)
                # crop and return:
                cropped = image[ y_offset: y_offset + self.crop_size[0],
                                 x_offset: x_offset + self.crop_size[1],
                               ]
                return cropped, boxes

        # this branch is for validation and test set:
        else:
            y_offset = (y_min + y_max) // 2
            if x_max_box + margin < self.crop_size[1]:
                x_offset = 0
            else:
                x_offset = (x_min + x_max) // 2

            # check if the current image (of validation/test set) has a normal volume (and not a biopsied one):
            if len(boxes["X"]) == 0:
                y_offset = (h // 2) - (self.crop_size[0] // 2)

                # checking the laterality of the image: {"R","L"}:
                img_laterality = "R" if image[h // 2, 0] == 0 else "L"

                if img_laterality == "L":
                    x_offset = 0
                else:
                    x_offset = w - self.crop_size[1]

                # crop and return:
                cropped = image[ y_offset: y_offset + self.crop_size[0],
                                 x_offset: x_offset + self.crop_size[1],
                                ]
                return cropped, boxes



        cropped = image[
            y_offset : y_offset + self.crop_size[0],
            x_offset : x_offset + self.crop_size[1],
        ]

        # don't let empty crop
        if np.max(cropped) == 0:
            y_offset = y_max // 2
            x_offset = 0
            cropped = image[
                y_offset : y_offset + self.crop_size[0],
                x_offset : x_offset + self.crop_size[1],
            ]

        boxes["X"] = [max(0, x - x_offset) for x in boxes["X"]]
        boxes["Y"] = [max(0, y - y_offset) for y in boxes["Y"]]
        # if cropped.shape != (1056,672):
        #     print("cropping in ", cropped.shape)

        return cropped, boxes

class RandomHorizFlip(object):

    def __init__(self, prob):
        assert isinstance(prob, (float, tuple))
        assert 0.0 < prob < 1.0
        self.prob = prob

    def __call__(self, sample):
        image, boxes = sample

        # for a given probability prob:
        # delete after:

        probb = np.random.random()
        # print("probb:",probb)
        if probb < self.prob:

            # flip the image horizontally:
            flipped = np.fliplr(image)

            length_key = len(boxes['X'])

            # iterate through all values of the dictionary boxes:
            for iter in range(length_key):

                # change the X coordinate of the bounding box:
                box_x = [int(x) for x in boxes["X"]][iter]
                box_w = [int(w) for w in boxes["Width"]][iter]
                elem = image.shape[1] - box_x
                # boxes["X"][iter] = elem
                boxes["X"][iter] = elem if elem > 0 else 0

        #  if randomly not flipping the image:
        else:
            flipped = image

        # print("flipped:", flipped.shape)
        # print()
        return flipped, boxes

class Normalize(object):

    def __init__(self, mean, std):
        assert isinstance(mean, (float, tuple))
        assert 0.0 < mean < 1.0
        self.mean = mean

        assert isinstance(std, (float, tuple))
        assert 0.0 < std < 1.0
        self.std = std

    def __call__(self, sample):
        image, boxes = sample

        # image = image.astype(np.float32)

        # # try to convert to range (0,1) first:
        image = image.astype(np.float32) / np.max(image)
        print("max value of image:", np.amax(image))
        print("min value of image:", np.amin(image))

        # convert numpy image to tensor image:
        image_tens = torch.from_numpy(image.copy())
        print("max value of image_norm:", torch.max(image_tens))
        print("min value of image_norm:", torch.min(image_tens))

        # normalize image:
        # image_norm = F.normalize(image_tens, self.mean, self.std)

        image_norm = (image_tens - self.mean) / self.std
        print("max value of image_norm:", torch.max(image_norm))
        print("min value of image_norm:", torch.min(image_norm))

        # image_norm2 = (image - self.mean) / self.std
        # print("max value of image_norm2:", max(image_norm2))
        # print("min value of image_norm:2", min(image_norm2))


        # x -= np.mean(x)  # the -= means can be read as x = x- np.mean(x)
        #
        # x /= np.std(x)


        # convert normalized tensor image to numpy image:
        image_np = image_norm.detach().cpu().numpy()

        return image_np, boxes