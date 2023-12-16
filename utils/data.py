import json
import math
import torch
import torchvision
import random
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.patches as patches

__all__ = ['VOCDataset', 'VOCRawTestDataset', 'load_data_voc']


categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


# Dataset adapt for Yolo format (divided into cells)
class VOCDataset(data.Dataset):
	def __init__(self, dataset, train=True):
		self.dataset = dataset
		self.train = train
	
	def __len__(self):
		return len(self.dataset)
	
	def __getitem__(self, idx):
		img, target = self.dataset[idx]
		obj = target['annotation']['object'][0]
		print(obj)
		xmin = int(obj['bndbox']['xmin'])
		ymin = int(obj['bndbox']['ymin'])
		xmax = int(obj['bndbox']['xmax'])
		ymax = int(obj['bndbox']['ymax'])
		name = obj['name']
		fig, ax = plt.subplots()
		ax.imshow(img.permute(1, 2, 0))
		rect = patches.Rectangle((xmin , ymin), (xmax - xmin), (ymax-ymin), linewidth = 5, edgecolor='r', facecolor='none')
		print(rect)
		ax.add_patch(rect)
		plt.show()
		
		if not isinstance(target['annotation']['object'], list):
			target['annotation']['object'] = [target['annotation']['object']]
		count = len(target['annotation']['object'])

		height, width = int(target['annotation']['size']['height']), int(target['annotation']['size']['width'])

		# resize to 448*448
		img = torchvision.transforms.functional.resize(img, (448, 448))

		# update labels from absolute to relative
		height, width = float(height), float(width)

		for i in range(count):
			obj = target['annotation']['object'][i]
			obj['bndbox']['xmin'] = float(obj['bndbox']['xmin']) / width
			obj['bndbox']['ymin'] = float(obj['bndbox']['ymin']) / height
			obj['bndbox']['xmax'] = float(obj['bndbox']['xmax']) / width
			obj['bndbox']['ymax'] = float(obj['bndbox']['ymax']) / height
		print('+'*10)

		obj = target['annotation']['object'][0]
		print(target['annotation']['object'][0])
		xmin = obj['bndbox']['xmin']
		ymin = obj['bndbox']['ymin']
		xmax = obj['bndbox']['xmax']
		ymax = obj['bndbox']['ymax']
		fig, ax = plt.subplots()
		ax.imshow(img.permute(1, 2, 0))
		rect = patches.Rectangle((xmin * 448.0, ymin * 448.0), (xmax - xmin) * 448.0, (ymax-ymin) * 448.0, linewidth = 5, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
		for i in range(7):
			for j in range(7):
				rect = patches.Rectangle((i * 448 / 7, j * 448 / 7), 448 / 7, 448 / 7, linewidth = 5, edgecolor='b', facecolor='none')
				ax.add_patch(rect)

		plt.show()

		# Label Encoding
		# [{'name': '', 'xmin': '', 'ymin': '', 'xmax': '', 'ymax': '', }, {}, {}, ...]
		# ==>
		# [x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  x, y  (relative to cell), width, height, 1 if exist (confidence),
		#  one-hot encoding of 20 categories]
		label = torch.zeros((7, 7, 30))
		for i in range(count):
			print('-------------------')
			print(obj)
			obj = target['annotation']['object'][i]
			xmin = obj['bndbox']['xmin']
			ymin = obj['bndbox']['ymin']
			xmax = obj['bndbox']['xmax']
			ymax = obj['bndbox']['ymax']
			name = obj['name']
			print(f'xmax: {xmax}. xmin {xmin}')

			if xmin == xmax or ymin == ymax:
				continue
			if xmin >= 1 or ymin >= 1 or xmax <= 0 or ymax <= 0:
				continue
			
			x = (xmin + xmax) / 2.0
			print(f'x: {x}, {xmin + xmax}')
			y = (ymin + ymax) / 2.0

			width = xmax - xmin
			height = ymax - ymin

			xidx = math.floor(x * 7.0)
			print(f'xidx: {xidx}')
			yidx = math.floor(y * 7.0)
			

			# According to the paper
			# if multiple objects exist in the same cell
			# pick the one with the largest area
			if label[yidx][xidx][4] == 1: # already have object
				if label[yidx][xidx][2] * label[yidx][xidx][3] < width * height:
					use_data = True
				else: use_data = False
			else: use_data = True

			if use_data:
				for offset in [0, 5]:
					# Transforming image relative coordinates to cell relative coordinates:
					# x - idx / 7.0 = x_cell / cell_count (7.0)
					# => x_cell = x * cell_count - idx = x * 7.0 - idx
					# y is the same
					label[yidx][xidx][0 + offset] = x * 7.0 - xidx
					print(f'x * 7.0 - xidx: {x * 7.0 - xidx}. Xmax: {xmax}, Xmin: {xmin}. x: {x}, xidx: {xidx}')
					label[yidx][xidx][1 + offset] = y * 7.0 - yidx
					print(f'y * 7.0 - yidx: {y * 7.0 - yidx}. Ymax: {ymax}, Ymin: {ymin}. y: {y}, xidx: {yidx}')
					exit()
					label[yidx][xidx][2 + offset] = width
					label[yidx][xidx][3 + offset] = height
					label[yidx][xidx][4 + offset] = 1
				label[yidx][xidx][10 + categories.index(name)] = 1

		return img, label


# Raw Dataset for testing mAP, Precision and Recall
# Target are 
class VOCRawTestDataset(data.Dataset):
	def __init__(self, dataset):
		self.dataset = dataset
	
	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		img, target = self.dataset[idx]

		if not isinstance(target['annotation']['object'], list):
			target['annotation']['object'] = [target['annotation']['object']]
		count = len(target['annotation']['object'])

		height, width = int(target['annotation']['size']['height']), int(target['annotation']['size']['width'])

		# resize to 448*448
		img = torchvision.transforms.functional.resize(img, (448, 448))

		# update labels from absolute to relative
		height, width = float(height), float(width)

		ret_targets = []

		for i in range(count):
			obj = target['annotation']['object'][i]

			ret_targets.append({
				'xmin': float(obj['bndbox']['xmin']) / width,
				'ymin': float(obj['bndbox']['ymin']) / height,
				'xmax': float(obj['bndbox']['xmax']) / width,
				'ymax': float(obj['bndbox']['ymax']) / height,
				'category': categories.index(obj['name']),
				'difficult': obj['difficult'] == '1',
			})
		
		return img, json.dumps(ret_targets)


def load_data_voc(batch_size, num_workers=0, persistent_workers=False, download=False, test_shuffle=True, trans = None):
	"""
	Loads the Pascal VOC dataset.
	:return: train_iter, test_iter, test_raw_iter
	"""
	# Load the dataset
	if trans == None:
		trans = [
			torchvision.transforms.ToTensor(),
		]
	trans = torchvision.transforms.Compose(trans)
	voc2007_trainval = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='trainval', download=download, transform=trans)
	voc2007_test = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2007', image_set='test', download=download, transform=trans)
	voc2012_train = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='train', download=download, transform=trans)
	voc2012_val = torchvision.datasets.VOCDetection(root='../data/VOCDetection/', year='2012', image_set='val', download=download, transform=trans)
	return (
		data.DataLoader(VOCDataset(data.ConcatDataset([voc2007_trainval, voc2007_test, voc2012_train]), train=True), 
			batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers), 
		data.DataLoader(VOCDataset(voc2012_val, train=False), 
			batch_size, shuffle=test_shuffle, num_workers=num_workers, persistent_workers=persistent_workers),
		data.DataLoader(VOCRawTestDataset(voc2012_val), 
			batch_size, shuffle=test_shuffle, num_workers=num_workers, persistent_workers=persistent_workers)
	)
