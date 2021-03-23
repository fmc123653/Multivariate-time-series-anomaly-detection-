import pandas as pd
import numpy as np
#  求解线性方程组
from scipy import linalg
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#temp为参数
def RBF_function(x,I,J,P,D,temp):#求两个数据节点相似度的函数
    ans=0
    for i in range(len(P)):#计算P中说明的变量
        for l in range(D):#计算长度
            k=P[i]
            ans+=(x[k][I+l]-x[k][J+l])**2
    ans/=(temp**2)
    #print(ans,np.exp(-ans))
    return np.exp(-ans)

def Train_model(S,c,d,n,iter_count):#训练模型
    for i in range(iter_count):
        c=d/n+(1-d)*np.matmul(S,c)
        c=(c-min(c))/(max(c)-min(c))
        #print(c.reshape([1,n]))
    return c

def  Frobenius_function(A,B):
    return sum(sum(A*B))

def Normalize_matrix(A):#矩阵数值归一化
    for i in range(len(A)):
        A[i]=(A[i]-min(A[i]))/(max(A[i])-min(A[i]))
    return A

def Aligned_kernel_matrix(data,N,P,y):#返回对齐的核矩阵
    Ky=np.zeros([N,N],dtype=float)
    for i in range(N):
        for j in range(N):
            Ky[i][j]=RBF_function(data,i,j,[y],1,0.2)
    X=[]
    for i in range(P):
        if i==y:
            continue
        X.append(data[i])
    X=np.array(X)
    A=[]#常数矩阵
    B=[]#系数矩阵
    for i in range(P-1):
        b=[]
        for j in range(P-1):
            b.append(Frobenius_function(np.matmul(X[i].reshape([N,1]),X[i].reshape([1,N])),np.matmul(X[j].reshape([N,1]),X[j].reshape([1,N]))))
        B.append(b)
        A.append(Frobenius_function(np.matmul(X[i].reshape([N,1]),X[i].reshape([1,N])),Ky))
    B=np.array(B)
    A=np.array(A)
    C=linalg.solve(B,A)#求解齐次线性方程结果
    K_temp=np.zeros([N,N],dtype=float)
    for i in range(P-1):
        K_temp+=(C[i]*np.matmul(X[i].reshape([N,1]),X[i].reshape([1,N])))
    K_temp=K_temp+np.abs(np.min(K_temp))
    return  K_temp
df=pd.read_csv('AirQualityUCI.tsv',sep=';')
P=4#选择4个变量
N=100#选择100个数据点
data=df.iloc[0:N,[2,3,4,6]].values.T#选取其中的2,3,4,6变量用来实验

for i in range(P):#数据归一化
    data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))

K_temp=np.identity(N)#初始化一个单位向量
Choose_P=[0,1,2,3]#选择你要进行识别的特定的目标变量
for i in range(len(Choose_P)):
    K_temp=np.matmul(K_temp,Aligned_kernel_matrix(data,N,P,Choose_P[i]))
S=Normalize_matrix(K_temp)#归一化
c=np.array([1]*N)
c=Train_model(S,c,0.7,N,50)#迭代次数为100
for i in range(len(data)):#打印原始数据图
    plt.subplot(2,2,i+1)
    plt.plot(data[i])
    plt.title('变量X'+str(i)+'数据图')
plt.figure()
plt.plot(c)
plt.title('异常检测结果图')
plt.show()