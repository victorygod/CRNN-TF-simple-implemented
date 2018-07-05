# CRNN-TF-simple-implemented

This is a simple tensorflow implementation of CRNN which can recognize Chinese.

Inspired by (& copied from) https://github.com/Belval/CRNN and https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow

## Reqiurements

*  Python 3.4+
*  tensorflow 1.4+
*  pillow
*  scipy
*  numpy
  
Anaconda Recommonded.

## Data prepare
The image files and the label files should have the same name. The label file is a csv file with the format in each line like this:
  bottom,left,right,top,value
  
Example label file:

    bottom,left,right,top,value
    109,137,342,59,中信银行
    118,1110,1206,69,账单
    239,50,172,217,科目： 20102
  
Put the image files and the label files in train/ dir

To run the inference procedure, put the image files and the annotation files in val/ dir. The annotation file should have the same format with the label file, except that there is no value entry at the end of each line.

## Run the model

Simply run python3 main.py

In the main.py file, you can choose which mode to use, either "train" or "infer". Just modify the code.
