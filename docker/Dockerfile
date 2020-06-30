FROM pytorch/pytorch:latest

COPY imitation_network.py .
COPY imitation_data.py .
COPY train.py .
COPY requirements.txt .


RUN pip install -r requirements.txt

RUN conda install pytorch==1.4.0 -c pytorch
