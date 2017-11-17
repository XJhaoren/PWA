# Part-based Weighting Aggregation (PWA)
Code for our AAAI2018 paper：
Unsupervised Part-based Weighting Aggregation of Deep Convolutional Features for Image Retrieval
Jian Xu, Cunzhao Shi, Chengzuo Qi, Chunheng Wang*, Baihua Xiao

NOTE:

tools:
</br>1.The python code is based on the python data science platform Anaconda2.
</br>2.The python code is tested on Windows by PyCharm.


data:
</br>3.You should uncompress the file "data\feature.rar" into current folder "data" firstly. The features of convolutional layer(Pool5 layer) of VGG16 for Oxford5k and Paris6k datasets are in path "data\feature". 
</br>4.The order of part detectors are in path "data\filter_select"
</br>5.The groundtruth for Oxford5k and Paris6k datasets are in path "data\gt_files"


code:
</br>6.Run evaluate.py, the mAP is printed.
</br>7.Run select_filter.py to get the order of part detectors according to variances. 
