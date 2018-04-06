# Cifar-10-Classifier
An image classifier trained on Cifar-10 dataset with 80.9% accuracy, inspired by assignment two of cs231n.



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

# Overall Architecture
The `~/graph` directory contains file that can visualize the graph on Tensorboard.
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/tensorboard.png)
<br>
<br>
The design of network is mainly motivated by (cnn-bn-relu)*n structure and AlexNet.
3 dropout layers were inserted to conquer the problem of overfitting which initial model previously suffered from, with dropout rate all set to be 0.5. This helped the model to generalze much better, as the accuracy finally raised by about 3%. 
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/arch.png)


# Predict
`Read_predict.py` is prepared for predicting on your own images, employing opencv and matplotlib packages to realize the visualization. You can change the paths to be your model and image at the start of the script:

```
#the model directory where .ckpt file was saved
model_path = './checkpoints/model.ckpt'
#input your image path
img_path = './test_images/horse.png'
```
A pre-trained model was kept for you in `~/checkpoints`. And you should see an output as below.
<br>
<br>
![Prediction on 'horse.png'](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/horse.png)

# Training and validation
Training and validation
The training process is executed in `cnn_tensorflow.py`. Personally I trained the model on Floydhub.com which is a website provides online training service with powerful GPUs, so as to achieve better performence of the model.
Here are some hyper-parameters you'll have to set before training, all of which you can find at the start of script with suggested initial values:
```
# hyper parameters
adam_beta1 = 0.88
adam_lr = 5e-4
train_batch_size = 128
plot_losses = True
```
To conduct training process, you can change `train_iter` and `last_checkpoint_dir` to start. That is, the code will then automatically execute restoring and saving operations in the `~/model_dir` directory as below(e.g. last_checkpoint_dir=50, train_iter=1):
<br>
<br>
![](https://github.com/AlanXia0118/Resource/blob/master/CIFAR-10-Classifier/model_dir.png) 
<br>
<br>
Hopefully steps above will help you realize more effiecient training process. Moreover, you can set `last_checkoint_dir` to be 0 to trigger an initialiaztion of whole graph to train a model from sketch.

For validation, just set the `train_iter` to be 0, and you should see a result like this on pre-trained model provided:
```
Validation
Epoch 1, Overall loss = 0.59 and accuracy of 0.809
test
Epoch 1, Overall loss = 0.597 and accuracy of 0.8
```
