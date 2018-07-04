# CRNN-TF-simple-implemented

This is a simple tensorflow implementation of CRNN which can recognize Chinese.

Inspired by (&copied) https://github.com/Belval/CRNN and https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow

## Reqiurements

*  Python 3.4+
*  tensorflow 1.4+
*  pillow
*  scipy
*  numpy
  
Annaconda Recommonded.

## Data prepare
The image files and the label files should have the same name. The label file is a csv file with the format in each line like this:
  bottom,left,right,top,value
  
Example label file:

    bottom,left,right,top,value
    109,137,342,59,中信银行
    118,1110,1206,69,账单
    239,50,172,217,科目： 20102
  
Put the image files and the label files in data/ dir

## Train the model

Simply run python3 main.py
