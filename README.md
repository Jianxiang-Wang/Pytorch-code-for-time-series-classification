# Pytorch-code-for-time-series-classification

Pytorch code for mutil-channel time series dataset. You can use this project to train LSTM to classify such data.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Introduction
For example, the shape of mutil-channel time series data should be like this (1000, 9000) in a csv file. It means that there are 1000 time series data. The first row of data is the label
of time series, such as (0, 1, 2, 3, ..., 9). We set there are 3 channels in the time series data, and it will be reshaped as (3000, 3).
