# PWA-Part-based-Weighting-Aggregation
Code for our AAAI2018 paper：
Unsupervised Part-based Weighting Aggregation of Deep Convolutional Features for Image Retrieval
Jian Xu, Cunzhao Shi, Chengzuo Qi, Chunheng Wang*, Baihua Xiao

NOTE:

tools:
1.The python code is based on the python data science platform Anaconda2.
2.The python code is tested on Windows by PyCharm.


data:
3.The features of convolutional layer(Pool5 layer) of VGG16 for Oxford5k and Paris6k datasets 
  are in path "PWA_code_and_data\data\feature".
  
4.The order of part detectors are in path "PWA_code_and_data\data\filter_select"

5.The groundtruth for Oxford5k and Paris6k datasets are in path "PWA_code_and_data\data\gt_files"


code:
6.Run evaluate.py, the mAP is printed.

7.Run select_filter.py to get the order of part detectors according to variances. 
