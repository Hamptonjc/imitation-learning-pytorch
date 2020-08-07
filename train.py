"""
Author: Jonathan Hampton
June 2020
"""

# Imports
import torch
import argparse
import pytorch_lightning as pl
from imitation_network import ImitationNetwork
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TensorBoardLogger

def main():

	""" Arguments """
	parser = argparse.ArgumentParser(description="Train imitation network")
	
	parser.add_argument(
		'-lr', '--learning_rate',
		type=float,
		default=0.0002,
		help="Learning rate for training network. Default = 0.0002")

	parser.add_argument(
		'-tb', '--train_batch_size',
		type=int,
		default=120,
		help="Batch size for training data. Default = 120")

	parser.add_argument(
		'-vb', '--val_batch_size',
		type=int,
		default=120,
		help="Batch size for validation data. Default = 120")

	parser.add_argument(
		'-t', '--train_data_dir',
		type=str,
		default="data-and-checkpoints/imitation_data/SeqTrain/",
		help="path to training data. Default == data-and-checkpoints/imitation_data/SeqTrain/")

	parser.add_argument(
		'-v', '--val_data_dir',
		type=str,
		default="data-and-checkpoints/imitation_data/SeqVal/",
		help="path to validation data. Default == data-and-checkpoints/imitation_data/SeqVal/")

	parser.add_argument(
		'-g', '--gpus',
		default=1,
		help="Number of GPUs. Default == 1.")

	parser.add_argument(
		'-n', '--run_name',
		type=str,
		default="",
		help="Name of training run for logging.")

	parser.add_argument(
		'-e', '--max_epochs',
		type=int,
		default=1,
		help="Maximum number of epochs: Default is 1.")

	parser.add_argument(
		'-ch', '--checkpoint_callback',
		type=bool,
		default=True,
		help="Save checkpoints of network. Default = True")

	parser.add_argument(
		'-fc', '--from_checkpoint',
		type=bool,
		default=False,
		help="begin training from checkpoint. Default = False")

	parser.add_argument(
		'-cp', '--checkpoint_path',
		type=str,
		default=None,
		help="path to checkpoint to resume training with. Default = None")

	parser.add_argument(
		'-es', '--early_stop_callback',
		type=bool,
		default=True,
		help="Enable early stopping. Default = True")

	parser.add_argument(
		'-pr', '--profiler',
		type=bool,
		default=False,
		help="Enable profiler. Default = False")

	parser.add_argument(
		'-dc', '--data_cache_size',
		type=int,
		default=100,
		help="Number of H5 files to be loaded at once in memory. Since there is training and validation datasets, size will be doubled. Default=100")

	parser.add_argument(
		'-l', '--loss_lambda',
		type=float,
		default=0.5,
		help="Value of lambda in loss function.")

	args = parser.parse_args()

	hparams = argparse.Namespace(**{'learning_rate':args.learning_rate,
		'train_batch_size': args.train_batch_size, 'val_batch_size': args.val_batch_size})


	""" Setup Network """

	checkpoint_callback = ModelCheckpoint(
		filepath='data-and-checkpoints/model_checkpoints/{epoch}-{val_loss:.2f}',
		save_last=True,
		monitor='val_loss')

	network = ImitationNetwork(
		data_cache_size = args.data_cache_size,
		lamb = args.loss_lambda,
		hparams=hparams,
		train_data_dir=args.train_data_dir,
		val_data_dir=args.val_data_dir)

	logger = TensorBoardLogger(
		"training_logs",
		name='Conditional Imitation Learning Network',
		version=args.run_name)

	if args.from_checkpoint:
		if args.checkpoint_path == None:
			Raise("Please specify path to checkpoint file (.ckpt)")
		else:
			trainer = pl.Trainer(
				resume_from_checkpoint=args.checkpoint_path,
				early_stop_callback=args.early_stop_callback,
				max_epochs=args.max_epochs,
				gpus=args.gpus,
				logger=logger,
				checkpoint_callback=checkpoint_callback,
				profiler=args.profiler)
	else:
		trainer = pl.Trainer(
			early_stop_callback=args.early_stop_callback,
			max_epochs=args.max_epochs,
			gpus=args.gpus,
			logger=logger,
			checkpoint_callback=checkpoint_callback,
			profiler=args.profiler)

	""" Train! :-) """
	trainer.fit(network)
	print("Training complete! Best checkpoint is", checkpoint_callback.best_model_path)


if __name__ == "__main__":
	
	main()
