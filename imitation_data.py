"""
Author: Jonathan Hampton
June 2020

"""

# Imports
import h5py
import torch
import random
import kornia
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils import data
from torchvision import transforms
from skimage.util import random_noise
from kornia.filters import gaussian_blur2d
import torchvision.transforms.functional as F


class ImitationDataset(data.Dataset):
	def __init__(self, data_dir, sort_by_command=False, data_cache_size = 5):
		super().__init__()
		self.sort_by_command = sort_by_command
		self.data_info = []
		self.data_cache = {}
		self.data_dir = data_dir
		self.data_cache_size = data_cache_size
		p = Path(data_dir)
		assert(p.is_dir())
		self.file_names	 = sorted(p.glob('*.h5'))
		if len(self.file_names) < 1:
			raise RuntimeError('No hdf5 data found')
		self.get_data_info(self.file_names)


	def imageTransform(self, image):
		image = np.array(image)
		random_transforms = transforms.RandomApply([
			RandomContrast(), RandomBrightness(), RandomGaussianBlur(),
			RandomSaltPepperNoise(), RandomGaussianNoise(), RandomReigonDropout()])
		image = random_transforms(image)
		return torch.tensor(image)

	def labelTransform(self, label):
		return torch.tensor(np.array(label))


	def __getitem__(self, index):
		data_point = self.get_data_point(index)
		image = self.imageTransform(data_point[0]).type(torch.FloatTensor)
		label = self.labelTransform(data_point[1]).type(torch.FloatTensor)
		input_data = [image, label[10], label[24]] # [img, speed, command]
		gt_action = torch.stack([label[0], label[1], label[2]]) # [steer angle, gas, brake]
		return input_data, gt_action

	def __len__(self):
		return len(self.data_info)

	def get_data_info(self, file_names):
		"""Walks through all h5 files and adds info of each data_point to data_info list"""
		if self.sort_by_command:
			dataset_idx = 0
			for file in tqdm(file_names, 'Getting data info'):
				try:
					with h5py.File(file,'r') as h5_file:
						for file_idx, data_point in enumerate(h5_file["targets"]):
							self.data_info.append([dataset_idx, file_idx, file, data_point[24]])
							dataset_idx += 1
				except:
					pass
			self.data_info = sorted(self.data_info, key = lambda x: x[3])
			for dataset_idx, data_point in enumerate(self.data_info):
				data_point[0] = dataset_idx

		else:
			dataset_idx = 0
			for file in tqdm(file_names, 'Getting data info'):
				try:
					with h5py.File(file,'r') as h5_file:
						for file_idx, data_point in enumerate(h5_file["targets"]):
							self.data_info.append([dataset_idx, file_idx, file])
							dataset_idx += 1
				except:
					pass

	def get_data_point_info(self, data_info, index):
		"""Gets data point info [dataset_idx, file_idx, file_name] from data_info list"""
		for data_point in data_info:
			if data_point[0] == index:
				return data_point

	def get_data_point(self, index):
		"""Returns requested data point from cache. if h5 file not in cache, loads it."""
		data_point_info = self.get_data_point_info(self.data_info, index)
		file_idx = data_point_info[1]
		file_name = data_point_info[2]
		if file_name not in self.data_cache:
			self.load_dataset(file_name)
		image = self.data_cache[file_name][0][file_idx]
		label = self.data_cache[file_name][1][file_idx]
		return [image, label]

	def load_dataset(self, file_name):
		""" Loads h5 file into cache"""
		with h5py.File(file_name, 'r') as file:
			images = np.array(file["rgb"][()])
			labels = np.array(file["targets"][()])
			self.add_to_cache([images, labels], file_name)
		if len(self.data_cache) > self.data_cache_size:
			remove_key = list(self.data_cache.keys())[0]
			del self.data_cache[remove_key]
			

	def add_to_cache(self, data, file_name):
		""" adds h5 dataset file to cache"""
		if file_name not in self.data_cache:
			self.data_cache[file_name] = data



""" ======= Custom Torch Transforms ======= """

class RandomContrast:
	"""
	applies random adjustment to image's contrast
	"""
	def __call__(self, imagearr):
		contrast_factor = random.uniform(0, 2)
		return np.array(F.adjust_contrast(Image.fromarray(imagearr), contrast_factor))

class RandomBrightness:
	"""
	applies random adjustment to image's brightness
	"""
	def __call__(self, imagearr):
		brightness_factor = random.uniform(0, 2)
		return np.array(F.adjust_brightness(Image.fromarray(imagearr), brightness_factor))

class RandomGaussianBlur:
	"""
	applies random Gaussian blur to image
	"""
	def __call__(self, image):
		random_kernel = random.randrange(25,35,2)
		random_kernel = (random_kernel, random_kernel)
		random_sigma = random.uniform(5,10)
		random_sigma = (random_sigma, random_sigma)		
		image = torch.unsqueeze(kornia.image_to_tensor(image).float(), dim=0)
		image = gaussian_blur2d(image, random_kernel, random_sigma)
		return kornia.tensor_to_image(image)

class RandomSaltPepperNoise:
	"""
	applies random salt and pepper noise to image
	"""
	def __call__(self, image):
		random_prob = random.random()
		return random_noise(image, mode='s&p', amount=random_prob)

class RandomGaussianNoise:
	"""
	applies random salt and pepper noise to image
	"""
	def __call__(self, image):
		return random_noise(image, mode='gaussian')

class RandomReigonDropout:
	""" Masks out a random set of squares in the image """
	def addRandomDropout(self, image):
	  rectangle_area = int(image.shape[0]*0.1)*int(image.shape[1]*0.1)
	  rectangle = np.reshape(np.array([0]*rectangle_area*3), (int(image.shape[0]*0.1),int(image.shape[1]*0.1),3))
	  x = random.randint(0,image.shape[0]-int(image.shape[0]*0.1))
	  y = random.randint(0,image.shape[1]-int(image.shape[1]*0.1))
	  image[x:x+int(image.shape[0]*0.1), y:y+int(image.shape[1]*0.1), :] = rectangle
	  return image

	def __call__(self, image):
		num_regions = random.randint(1, 20)
		for _ in range(num_regions):
			image = self.addRandomDropout(image)
		return image

