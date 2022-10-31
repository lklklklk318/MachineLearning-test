
# coding: utf-8

# In[15]:


#3.5
import numpy as np  
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import matplotlib.pyplot as plt

# 下载csv文件
dataset = np.loadtxt('D:/MachineLearning/watermelon3_0a.csv', delimiter=",")
#输出数据
print(dataset)
#密度 含糖率提取出来
X = dataset[:, 1:3]
#是否好瓜坏瓜提取出来
y = dataset[:, 3]
#好瓜
goodData=dataset[:8]
#坏瓜
badData=dataset[8:]
# 数量 17行 2列
m, n = np.shape(X)
'''
LDA via sklearn
'''

# XY的各百分之五十作为测试机和训练集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=0)
# 线性判别分析 lsqr：最小平方差 fix：训练模型
lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X_train, y_train)
# 用模型进行预测 返回预测值
y_pred = lda_model.predict(X_test)
#混淆矩阵 实际目标值，预测值
print('混淆矩阵为：')
print(metrics.confusion_matrix(y_test, y_pred))
#准确率
print('准确率为：')
print(metrics.classification_report(y_test, y_pred))
#画图
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
"""
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='b', s=100, label='bad')
"""

plt.scatter(goodData[:,1], goodData[:,2], marker='o', color='g', s=100, label='good')
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
plt.legend(loc='upper right')
 
 
'''
implementation of LDA based on self-coding
'''
u = [[badData[:,1].mean(),badData[:,2].mean()],[goodData[:,1].mean(),goodData[:,2].mean()]]
u=np.matrix(u)
 
Sw = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)  # row -> cloumn vector
    if y[i] == 0: u_tmp = u[0].reshape(n,1)
    if y[i] == 1: u_tmp = u[1].reshape(n,1)
    Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )

U, sigma, V= np.linalg.svd(Sw)

Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T

w = np.dot( Sw_inv, (u[0] - u[1]).reshape(n,1) )  # here we use a**-1 to get the inverse of a ndarray
print(w)

f3 = plt.figure(3)
plt.xlim( 0, 1 )
plt.ylim( 0, 0.7 )
 
plt.title('watermelon_3a - LDA')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=10, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=10, label = 'good')
plt.legend(loc = 'upper right')
 
k=w[1,0]/w[0,0]
plt.plot([-1,1], [-k, k])
 
for i in range(m):
    curX=(k*X[i,1]+X[i,0])/(1+k*k)
    if y[i]==0:plt.plot(curX,k*curX,"ko",markersize=3)
    else:plt.plot(curX,k*curX,"go",markersize=3)
    plt.plot([curX,X[i,0]],[k*curX,X[i,1]],"c--",linewidth=0.3)
 
plt.show()

