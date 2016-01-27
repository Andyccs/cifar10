# cifar10

CZ4041 Machine Learning Assignment

# Getting Started

You need to login to Kaggle and download the dataset from [Kaggle Competition](https://www.kaggle.com/c/cifar-10). Next, you should put all the downloaded files, i.e. `sampleSubmission.csv`, `test.7z`, `train.7z`, and `trainLabels.csv`, to `dataset` folder. This folder will not be checked in to Git version control. To extract the `.7z` files, you need to [download 7zip in the official website](http://www.7-zip.org/download.html), or use brew to install it if you're using MAC OS X:

```Shell
brew install p7zip
```

After downloading 7zip, you should extract the `.7z` files to `testdata` folder and `traindata` folder. Similary, these folders will not be checked in. The extraction should take around 5 to 10 minutes. You should be able to extract these folders easily in Windows (I suppose). You can use the following command if you're using MAC OS X or Linux:

```Shell
7z e -y -otraindata dataset/train.7z
7z e -y -otestdata dataset/test.7z
```

# References

1. [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html)
2. [Classification datasets results](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html)
3. [Kaggle CIFAR-10 Object Recognition in Images](https://www.kaggle.com/c/cifar-10)
4. [Udacity Deep Learning Course](https://www.kaggle.com/c/cifar-10)