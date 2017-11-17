#coding:utf-8 允许中文注释
import os
import numpy as np

# # oxford-vgg-512
# sum_matrix=[]
# file_load = '../data/feature/oxford_dataset_feature'
# directorys= os.listdir(file_load)
# for directory in directorys:
#      feature_map= np.load(os.path.join(file_load,directory))
#      sum=(feature_map.sum(1)).sum(1)
#      # sum_matix.append(sum)
#      if sum_matrix==[]:
#          sum_matrix=sum
#      else:
#          sum_matrix=np.vstack((sum_matrix, sum))
#
# average=sum_matrix.sum(0)/sum_matrix.shape[0]
# variance=sum_matrix-average
# variance=variance*variance
# variance=variance.sum(0)
#
# dic_variance = dict(zip(variance, range(variance.shape[0])))
# dic_variance_sorted = dic_variance.items()
# dic_variance_sorted.sort(reverse=True)
#
# select_num=[value for key, value in dic_variance_sorted]
# select_save='../data/filter_select/select_num_oxford.npy'
# np.save(select_save,select_num)




# paris-vgg-512
sum_matrix=[]
file_load = '../data/feature/paris_dataset_feature'
directorys= os.listdir(file_load)
for directory in directorys:
     feature_map= np.load(os.path.join(file_load,directory))
     sum=(feature_map.sum(1)).sum(1)
     # sum_matix.append(sum)
     if sum_matrix==[]:
         sum_matrix=sum
     else:
         sum_matrix=np.vstack((sum_matrix, sum))

average=sum_matrix.sum(0)/sum_matrix.shape[0]
variance=sum_matrix-average
variance=variance*variance
variance=variance.sum(0)

dic_variance = dict(zip(variance, range(variance.shape[0])))
dic_variance_sorted = dic_variance.items()
dic_variance_sorted.sort(reverse=True)

select_num=[value for key, value in dic_variance_sorted]
select_save='../data/filter_select/select_num_paris.npy'
np.save(select_save,select_num)