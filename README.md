# Conditional Imitation Learning at CARLA (Unofficial PyTorch Implementation)

This is an unofficial PyTorch implementation of [End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410). All credit to the original researchers.

## Dataset

The dataset can be downloaded [here](https://drive.google.com/file/d/1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY/view). It is 24GB of HDF5
files. imitation_data.py is a custom Torch dataset class which handles and preprocesses the dataset.

## Setup 
### For Training:
- Clone repo
- put dataset into data-and-checkpoints/imitation_data
- Run docker-compose file to build image and start container.
- From within the container, run train.py.

Training logs will output to host training logs folder for easy Tensorboard access :-)

Note, the Docker container requires Nvidia Docker runtime.

### Agent:
- Specify network checkpoint in Agent class init method.
- May need to rewrite certain Carla imports.


## Future Work

- [ ] Implement branch-specific backpropagation.
