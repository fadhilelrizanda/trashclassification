# Trash Classification Using Transfer Learning EfficientNet

The repository is about building model based on efficientNet to perform trash classification from 6 classes. The model separated to using class weight and non class weight for the training

## Instalation

All libraries to running the code are in `requirements.txt`. To install it just perform
`pip install -r requirements.txt`

## Download Dataset

The dataset can be found in [https://huggingface.co/datasets/garythung/trashnet](trashnet). To get the data set we can use. `git clone https://huggingface.co/datasets/garythung/trashnet `

## Training the model

The model separated to using class weight and non class weight for the training. To train the model (without link it to wand.db) just perform it using `python train.py`
