
# Imports
import torch
import argparse
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from carla_084.PythonClient.carla.agent import Agent
from carla_084.PythonClient.carla.carla_server_pb2 import Control
from imitation_learning.imitation_network import ImitationNetwork
import cv2
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)



class ImitationAgent(Agent):

	def __init__(self, image_cut=[115, 510]):
		Agent.__init__(self)
		self.image_cut = image_cut
		hparams = argparse.Namespace(**{'learning_rate':1,
			'train_batch_size': 1, 'val_batch_size': 1})
		self.cil_net = ImitationNetwork.load_from_checkpoint('./data-and-checkpoints/model_checkpoints/last.ckpt')
		self.cil_net.freeze()
		self.input_image_size = (200, 88)

	def run_step(self, measurements, sensor_data, directions, target):
		control = self.compute_action(sensor_data['CameraRGB'].data,
			measurements.player_measurements.forward_speed, directions)
		return control

	def compute_action(self, rgb_image, speed, direction=None):
		rgb_image = rgb_image #[self.image_cut[0]:self.image_cut[1], :]
		rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
		image_input = np.array(Image.fromarray(rgb_image).resize(size=self.input_image_size))
		#image_input = image_input.astype(np.float32)
		cv2.imshow("image_input", image_input)
		cv2.waitKey(1)
		image_input = np.multiply(image_input, 1.0 / 255.0)
		logging.info("direction:", direction)

		steer, acc, brake = self.control_function(image_input, speed, direction)

		if brake < 0.1:
			brake = 0.0

		if acc > brake:
			brake = 0.0

		if speed > 10.0 and brake == 0.0:
			acc = 0.0

		control = Control()
		control.steer = steer
		control.throttle = acc
		control.brake = brake
		control.hand_brake = 0
		control.reverse = 0
		return control

	def control_function(self, image_input, speed, direction):
		input_data = [torch.tensor(image_input).unsqueeze(dim=0),
		torch.tensor(speed).unsqueeze(dim=0), torch.tensor(direction).unsqueeze(dim=0)]
		output = self.cil_net(input_data)
		steer = output[0][0]
		acc = output[0][1]
		brake = output[0][2]
		return steer, acc, abs(brake)
