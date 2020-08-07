"""
Author: Jonathan Hampton
June 2020

"""

# Imports
import os
import torch
import argparse
import torch.nn as nn
import imitation_data
import pytorch_lightning as pl
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader


class ImitationNetwork(pl.LightningModule):

	def __init__(self, data_cache_size=100, lamb=0.5, hparams=argparse.Namespace(
		**{'learning_rate':1,'train_batch_size': 1, 'val_batch_size': 1}),
		train_data_dir=None, val_data_dir=None):
		super().__init__()
		torch.cuda.empty_cache()
		self.train_data_dir = train_data_dir
		self.val_data_dir = val_data_dir
		self.learning_rate = hparams.learning_rate
		self.train_batch_size = hparams.train_batch_size
		self.val_batch_size = hparams.val_batch_size
		self.data_cache_size = data_cache_size
		self.lamb = lamb

		"""layers"""
		self.imageModule = nn.Sequential(*self.get_image_module())
		self.measurementModule = nn.Sequential(*self.get_measurement_module())
		self.jointSensoryModule = nn.Sequential(*self.get_joint_sensory_module())
		self.follow_lane_branch = nn.Sequential(*self.get_branch_module())
		self.left_branch = nn.Sequential(*self.get_branch_module())
		self.right_branch = nn.Sequential(*self.get_branch_module())
		self.straight_branch = nn.Sequential(*self.get_branch_module())
		self.module_list = [self.imageModule, self.measurementModule,\
		self.jointSensoryModule, self.follow_lane_branch, self.left_branch,\
		self.right_branch, self.straight_branch]
		
		"""Initialize weights with xavier initialization"""
		for module in self.module_list:
			module.apply(self.init_weights)

	def init_weights(self, m):
		if type(m) == nn.Linear:
			torch.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(0.1)

	def fc(self, in_features, out_features):
		return nn.Linear(in_features, out_features, bias=True)

	def convBlock(self, input_channels, output_channels, kernel_size, stride, flatten_output=False):
		conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride)
		bn = nn.BatchNorm2d(output_channels)
		dropout = nn.Dropout2d(p=0.2, inplace=True)
		relu = nn.ReLU()
		if flatten_output:
			flatten = nn.Flatten()
			return [conv, bn, dropout, relu, flatten]
		else:
			return [conv, bn, dropout, relu]

	def fcBlock(self, in_features, out_features):
		fc = self.fc(in_features, out_features)
		dropout = nn.Dropout(p=0.5, inplace=True)
		relu = nn.ReLU()
		return [fc, dropout, relu]

	def get_image_module(self):
		imageModule = []
		imageModule.extend(
		self.convBlock(3, 32, 5, 2) +\
		self.convBlock(32, 32, 3, 1) +\
		self.convBlock(32, 64, 3, 2) +\
		self.convBlock(64, 64, 3, 1) +\
		self.convBlock(64, 128, 3, 2) +\
		self.convBlock(128, 128, 3, 1) +\
		self.convBlock(128, 256, 3, 1) +\
		self.convBlock(256, 256, 3, 1, flatten_output=True) +\
		self.fcBlock(8192, 512)+\
		self.fcBlock(512, 512))
		return imageModule
	
	def get_measurement_module(self):
		measurementModule = []
		measurementModule.extend(
			self.fcBlock(1, 128) +\
			self.fcBlock(128, 128))
		return measurementModule

	def get_joint_sensory_module(self):
		return self.fcBlock(640, 512)

	def get_branch_module(self):
		branchModule = []
		branchModule.extend(
			self.fcBlock(512, 256) +\
			self.fcBlock(256, 256) +\
			[self.fc(256,3)]) # 3: steer angle, throttle, brake
		return branchModule

	def gated_branch_function(self, j_batch, control_batch):
		batch_output = []
		for j, control in zip(j_batch, control_batch):
			control = int(control.item())
			s = torch.cuda.Stream()
			""" Branches """
			if control == 2 or control == 0:
				with torch.cuda.stream(s):
					output = self.follow_lane_branch(j)
					batch_output.append(output)
			elif control == 3:
				with torch.cuda.stream(s):
					output = self.left_branch(j)
					batch_output.append(output)
			elif control == 4:
				with torch.cuda.stream(s):
					output = self.right_branch(j)
					batch_output.append(output)
			elif control == 5:
				with torch.cuda.stream(s):
					output = self.straight_branch(j)
					batch_output.append(output)
		torch.cuda.synchronize()
		return torch.stack(batch_output)


	def forward(self, input_data):
		''' Define variables '''
		input_image = input_data[0].permute(0,3,2,1).float()
		input_speed = input_data[1].unsqueeze(dim=1)
		control = input_data[2]

		''' Pass input data into network '''
		imageOutput = self.imageModule(input_image)
		speedOutput = self.measurementModule(input_speed)

		''' Joint sensory '''
		j = torch.cat([imageOutput, speedOutput], 1)
		j = self.jointSensoryModule(j)

		''' Branches '''
		output = self.gated_branch_function(j, control)
		return output

	def custom_loss(self, model_output, label, lamb=self.lamb):
		steer_angle = model_output[:,0]
		steer_gt = label[:,0]
		throttle = model_output[:,1]
		throttle_gt = label[:,1]
		brake = model_output[:,2]
		brake_gt = label[:,2]
		acc = throttle - brake
		acc_gt = throttle_gt - brake_gt
		loss = torch.norm(steer_angle - steer_gt)**2 + lamb*torch.norm(acc - acc_gt)**2
		return torch.sum(loss)

	def configure_optimizers(self):
		optim = torch.optim.Adam(self.parameters())
		return optim

	def training_step(self, train_batch, batch_idx):

		input_data, label = train_batch
		model_output = self.forward(input_data)
		loss = self.custom_loss(model_output, label)
		train_logs = {'training_loss': loss}
		return {'loss': loss, 'log': train_logs}

	def validation_step(self, val_batch, batch_idx):
		input_data, label = val_batch
		model_output = self.forward(input_data)		
		loss = self.custom_loss(model_output, label)
		val_logs = {'validation_loss': loss}
		return {'val_loss': loss, 'log': val_logs}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		tensorboard_logs = {'average_validation_loss': avg_loss}
		return {'val_loss': avg_loss, 'log': tensorboard_logs}

	def train_dataloader(self):
		print("Preparing training data...")
		self.train_dataset = imitation_data.ImitationDataset(data_dir=self.train_data_dir, sort_by_command=True, data_cache_size=self.data_cache_size)
		print("Training dataset prepared!")
		return DataLoader(self.train_dataset,batch_size=self.train_batch_size,
			num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

	def val_dataloader(self):
		print("Preparing validation data...")
		self.val_dataset = imitation_data.ImitationDataset(data_dir=self.val_data_dir, data_cache_size=self.data_cache_size)
		print("validation dataset prepared!")
		mp.set_start_method('spawn', force=True)
		return DataLoader(self.val_dataset,batch_size=self.train_batch_size,
			num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

