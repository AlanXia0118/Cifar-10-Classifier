# Cifar-10-Classifier
An image classifier trained on Cifar-10 dataset, inspired by assignment two of cs231n.



# Enviroment
Packages utilized in this project including:  
* python 3.5.4  
* tensorflow 1.7  
* opencv 3.1.0  
* matplotlib 2.0.2  
* numpy 1.12.1

All of which you can easily get access to on Anaconda.

# Dataset
The current model was trained on CIFAR-10 dataset which consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. For more details please refer to http://www.cs.toronto.edu/~kriz/cifar.html.  
Here we use `~/load_data/data_utils.py` to extract the data for training, validation and test. CIFAR-10 dataset has already been divided into several batches in the following directory, which you can take as input directly:  
```~/load_data/datasets/cifar-10-batches-py```

# Predict
`Read_predict.py` is prepared for predicting with your own images, employing opencv and matplotlib packages to realize the visualization. You can change the paths to be your model and image at the start of the script:

```
#the model directory where .ckpt file was saved
model_path = './checkpoints/model.ckpt'
#input your image path
img_path = './test_images/horse.png'
```
And you should see a result as below.
![Prediction on 'horse.png'](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/read_predict.png)
