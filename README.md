# cifar10

CZ4041 Machine Learning Assignment

# Getting Started

Let's start by installing all required modules for this project. We install TensorFlow by following [the guide from official website](https://www.tensorflow.org/versions/0.6.0/get_started/os_setup.html). Then, we install all other modules using the following commands:

```Shell
pip install -r requirements.txt
```

Next, you need to login to Kaggle and download the dataset from [Kaggle Competition](https://www.kaggle.com/c/cifar-10). Next, you should put all the downloaded files, i.e. `sampleSubmission.csv`, `test.7z`, `train.7z`, and `trainLabels.csv`, to `dataset` folder. This folder will not be checked in to Git version control. To extract the `.7z` files, you need to [download 7zip in the official website](http://www.7-zip.org/download.html), or use brew to install it if you're using MAC OS X:

```Shell
brew install p7zip

# To check whether you finished your download
python src/check_dataset.py
```

After downloading 7zip, we extract the `.7z` files to `testdata` folder and `traindata` folder. Similary, these folders will not be checked in. You should be able to extract these folders easily in Windows. You can use the following command if you're using MAC OS X or Linux:

```Shell
# The extraction should take around 1 to 2 hour
7z e -y -otraindata dataset/train.7z
7z e -y -otestdata dataset/test.7z
```

After extracting the files, we might want to take a look at the data. The following command will open 10 random images from `traindata` folder:

```Shell
python src/show_random_picture.py
```

Now we want to change all images to matrices, so that we can feed these matrices to our awesome classifier. We have have wrote a script to change an image to a three dimensional `32 X 32 X 3` matrix, so the whole dataset is represented in a four dimensional matrix (with an additional dimension for the number of images). The script will also turn train labels in .csv file format to a one dimensional array. We use the following command to convert images to matrices: 

```Shell
# The conversion will take around 5-10 minutes
# After the conversion, we will the two more .pickle file at root directory
# cifar10_test.pickle should be around 3.69GB
# cifar10_train.pickle should be around 614.8MB
python src/images_to_matrices.py
```

# Contribution

Please follow the guide in [CONTRIBUTING.md](CONTRIBUTING.md)

# References

1. [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
2. [Classification datasets results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)
3. [Kaggle CIFAR-10 Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
4. [Udacity Deep Learning Course](https://www.kaggle.com/c/cifar-10)