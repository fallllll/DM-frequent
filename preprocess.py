import pandas as pd
import numpy as np
from fancyimpute import KNN, SoftImpute, IterativeImputer, BiScaler
from collections import Counter
import matplotlib.pyplot as plt


# 用最高频率值来填补缺失值
def na_max(path):
    wine_data = pd.read_csv(path, header=0, index_col=0, engine='python', encoding='utf-8')
    wine_data = wine_data.values
    max_time = []  # 每个属性最大频数的值
    # 确定每个属性最大频数的值
    for cl in range(wine_data.shape[1]):
        counter = Counter(wine_data[:, cl])
        counter = counter.most_common()  # 排序，返回类型为list，list的每个元素为内容和频数
        if counter[0][0] == counter[0][0]:  # 如果最大频数不为空值
            max_time.append(counter[0][0])
            # print(str(counter[0][0])+'非空')
        else:  # 如果最大频数为空值
            max_time.append(counter[1][0])
            # print(str(counter[1][0]) + '空')
        # print(list(counter.keys())[0])

    # 对每个属性的空值进行替换
    wine_max = pd.DataFrame(wine_data)
    for cl in range(wine_data.shape[1]):
        wine_max[cl] = wine_max[cl].fillna(max_time[cl])
        # print(max_time[cl])
    wine_max.to_csv('C:/Users/xue/Desktop/课程_研一下/数据挖掘/课后作业/4/wine-reviews/fill_max.csv')


    return wine_max


path = 'C:/Users/xue/Desktop/课程_研一下/数据挖掘/课后作业/4/wine-reviews/winemag-data_first150k.csv'
wine_max = na_max(path)  # 用最高频率值来填补缺失值
