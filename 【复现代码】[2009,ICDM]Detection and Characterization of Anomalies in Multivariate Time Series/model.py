description='\
--------------------------------------模型代码使用说明------------------------------------------------\n\
|                      导入model.py文件直接调用model.Run_model()                                     |\n\
---------------------------------------------------------------------------------------------------\n\
|                                                                                                 |\n\
|       c=model.Run_model(data,D,Choose_P,training_step,temp,d,detection_key,local_K)             |\n\
|                                                                                                 |\n\
|-------------------------------------------------------------------------------------------------|\n\
|data:输入的数据，为二维数据矩阵，列向为变量，横向为每个变量对应的数据，如果数据有P条，每条有N个数据点，data大小为[P,N]|\n\
|D:为int型数据，针对区间异常检测时的参数，为检测的区间的大小，需自行调整                                       |\n\
|Choose_P:为列表，针对特定目标检测时的参数，列表内为需要特定识别的变量下标，如果为通用异常检测模式该参数输入空列表[]   |\n\
|training_step:迭代运算的次数                                                                       |\n\
|temp:浮点数，相似度计算参数                                                                          |\n\
|d:浮点数，用于计算迭代公式的参数d/n+(1-d)Sc                                                            |\n\
|detection_key:整型，识别的类型，如果为1进行点对点异常检测，如果为2进行区间异常检测，如果为3进行局部异常检测        |\n\
|local_K:整型，针对局部异常检测时的局部宽度                                                             |\n\
|------------------------------------------------------------------------------------------------|\n\
'

import numpy as np
from multiprocessing import Manager
from multiprocessing import Process
#  求解线性方程组
from scipy import linalg
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#temp为参数
def RBF_function(x,I,J,P,D,temp):#求两个数据节点相似度的函数
    res=0
    node_nums=len(x[0])
    for i in range(len(P)):#计算P中说明的变量
        ans=0
        for l in range(D):#计算长度
            if I+l>=node_nums or J+l>=node_nums:
                break
            k=P[i]
            ans+=(x[k][I+l]-x[k][J+l])**2
        #ans/=D
        #res+=ans
        res+=(ans/np.var(x[P[i]]))
    return np.exp(-res)

def Train_model(S,c,d,n,training_step):#训练模型
    for i in range(training_step):
        c=d/n+(1-d)*np.matmul(S,c)
        c=(c-min(c))/(max(c)-min(c))
        #print(c.reshape([1,n]))
    return c

def  Frobenius_function(A,B):
    return sum(sum(A*B))

def Local_detection(N,K,local_K):#区部异常检测，删除边
    for i in range(N):
        for j in range(N):
            if abs(j-i)>local_K:
                K[i][j]=0.0#删除节点i到节点j之间的边
    return K

def Normalize_matrix(A):#针对矩阵数值归一化，要保证对称性
    return (A-np.min(A))/(np.max(A)-np.min(A))
def Description():#描述函数
    print(description)
'''
def Aligned_kernel_matrix(data,N,D,P,y,temp):#返回对齐的核矩阵
    Ky=np.zeros([N,N],dtype=float)
    for i in range(N):
        for j in range(N):
            Ky[i][j]=RBF_function(data,i,j,[y],D,temp)
    X=[]
    for i in range(P):
        if i==y:
            continue
        tx=[]
        #for j in range(0,N):
        #    tx.append(np.mean(data[i][j:j+D]))
        #X.append(tx)
        X.append(data[i][0:N])
        #print(data[i][0:N])
        #print(i,'*'*100)
    X=np.array(X)

    #print(np.matmul(X[4].reshape([N,1]),X[4].reshape([1,N])))
    #print('*'*100)
    #print(X[4])
    #print(data[5])
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
    #print(B)
    #print(np.linalg.det(B))
    C=linalg.solve(B,A)#求解齐次线性方程结果
    K_temp=np.zeros([N,N],dtype=float)
    for i in range(P-1):
        K_temp+=(C[i]*np.matmul(X[i].reshape([N,1]),X[i].reshape([1,N])))
    K_temp=K_temp+np.abs(np.min(K_temp))
    return  K_temp
'''
def Aligned_kernel_matrix(index,data,N,D,P,y,temp,detection_key,local_K,common_data):#返回对齐的核矩阵
    print('进程'+str(index+1)+'开始运算......')
    Kx=np.zeros([N,N],dtype=float)
    Ky=np.zeros([N,N],dtype=float)
    for i in range(N):
        for j in range(N):
            Ky[i][j]=RBF_function(data,i,j,[y],D,temp)

    if detection_key==3:#进行局部异常检测处理
        Ky=Local_detection(N,Ky,local_K)

    X=[]
    for i in range(P):
        if i==y:
            continue
        X.append(data[i])
    for i in range(N):
        for j in range(N):
            Kx[i][j]=RBF_function(X,i,j,np.arange(0,P-1),D,temp)

    if detection_key==3:#进行局部异常检测处理
        Kx=Local_detection(N,Kx,local_K)

    Kx=Normalize_matrix(Kx)#矩阵归一化，保证对称性，不然特征值可能为虚数

    #print(np.linalg.det(Kx))

    w, v = np.linalg.eig(Kx)
    w=w.real#取实数部分
    v=v.real#取实数部分
    #print(type(w[0]))
    #print(v)
    '''
    #Test...............
    K_res=np.zeros([N,N],dtype=float)
    for i in range(len(w)):
        K_res+=(w[i]*np.matmul(v[:,i].reshape([N,1]),v[:,i].reshape([1,N])))
    return K_res
    '''

    C=[]
    for i in range(N):
        C.append(w[i]+(Frobenius_function(np.matmul(v[:,i].reshape([N,1]),v[:,i].reshape([1,N])),Ky))/2)
    K_temp = np.zeros([N, N], dtype=float)

    for i in range(N):
        K_temp+=(C[i]*np.matmul(v[:,i].reshape([N,1]),v[:,i].reshape([1,N])))
    common_data.append(K_temp)#把运算的结果储存下来
    #return K_temp

def Normalize_function(data):#给一般的数组归一化
    new_data=np.zeros([data.shape[0],data.shape[1]],dtype=float)
    for i in range(len(data)):
        if max(data[i])-min(data[i])<=10**-8:
            continue
        new_data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
    return new_data

def del_zero_matrix(data):#删除无用变量
    X=[]
    for i in range(len(data)):
        if max(data[i])-min(data[i])<=10**-8:
            continue
        X.append(data[i])
    return np.array(X)



def Run_model(data,D,Choose_P,training_step,temp,d,detection_key,local_K):
    if detection_key==1:
        D=1
        print('开始点对点异常检测方式...')
        if len(Choose_P)==0:
            print('检测目标：通用!')
        else:
            print('检测目标：特定！')
    if detection_key==2:
        print('开始区间异常检测方式...')
        if len(Choose_P)==0:
            print('检测目标：通用!')
        else:
            print('检测目标：特定！')
    if detection_key==3:
        D=1
        print('开始局部异常检测方式...')
        if len(Choose_P)==0:
            print('检测目标：通用!')
        else:
            print('检测目标：特定！')
    data = Normalize_function(data)  # 数据归一化

    data = del_zero_matrix(data)  # 删除无用变量
    N=len(data[0])#数据长度，节点个数
    K_temp = np.identity(N)  # 初始化一个单位向量
    P=len(data)#变量数目

    if len(Choose_P)==0:#如果是空的表示没有选择，默认选择全部
        Choose_P=np.arange(0,P)#选择的要重点关注的变量信息

    print('检测到'+str(len(Choose_P))+'个目标变量，开启'+str(len(Choose_P))+'个进程...')

    jobs = []#装进程
    common_data = Manager().list()  # 这里是声明一个列表的共享变量
    for i in range(len(Choose_P)):#开启对应的进程
        p=Process(target=Aligned_kernel_matrix,args=(i,data,N,D,P,Choose_P[i],temp,detection_key,local_K,common_data))#共享common_data这个变量
        jobs.append(p)
        p.start()#开始进程

    for proc in jobs:
        proc.join()#利用阻塞等待所有进程结束才往下执行

    K_temp = np.identity(N)  # 初始化一个单位向量
    for i in range(len(Choose_P)):
        K_temp=np.matmul(K_temp,common_data[i])#全部乘起来

    #for i in range(len(Choose_P)):
    #    print('step...',i)
    #    K_temp = np.matmul(K_temp, Aligned_kernel_matrix(data, N,D, P, Choose_P[i],temp))

    S = Normalize_matrix(K_temp)  # 归一化

    print('运算结束..........')

    c = np.array([1] * N)
    c = Train_model(S, c,d, N, training_step)  # 迭代次数为100

    #print(Normalize_function(K_total))
    #print(S)

    return c
