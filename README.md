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
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/cifar-10.png)
<br>
<br>
To load the data for training, validation, and test, firstly run the following command in the `~/load_data/datasets` directory:
```
sh get_datasets.sh
```
Then the CIFAR-10 dataset will be divided into several batches and saved in the `~/load_data/datasets/cifar-10-batches-py` directory. Later, we call the `load_CIFAR10()` function to extract data batches which you can take directly as input.


# Predict
`Read_predict.py` is prepared for predicting on your own images, employing opencv and matplotlib packages to realize the visualization. You can change the paths to be your model and image at the start of the script:

```
#the model directory where .ckpt file was saved
model_path = './checkpoints/model.ckpt'
#input your image path
img_path = './test_images/horse.png'
```
And you should see an output as below.
<br>
<br>
![Prediction on 'horse.png'](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/horse.png)
