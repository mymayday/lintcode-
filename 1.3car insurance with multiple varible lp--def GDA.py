import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

#数据的导入
data_train=pd.read_csv('train.csv')           #读取训练数据
data_test=pd.read_csv('test.csv')             #读取测试数据

Y=data_train.Score                       #读取训练数据中的Score字段         输出

#对数据进行预处理
data_train.dropna(axis=0,how='any',subset=['Score'],inplace=True)         #删除Score为空值的数据行
                                   #.dropna的用法：DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
                                   #功能：根据各标签的值中是否存在缺失数据对轴标签进行过滤，可通过阈值调节对缺失值的容忍度
                                   #参数：axis : {0 or ‘index’, 1 or ‘columns’},或 tuple/list 
                                   #how : {‘any’, ‘all’}    any : 如果存在任何NA值，则放弃该标签   all : 如果所有的值都为NA值，则放弃该标签
                                   #thresh : int, 默认值 None   　int value ：要求每排至少N个非NA值　　
                                   #subset : 类似数组
                                   #inplace : boolean, 默认值 False   如果为True，则进行操作并返回None。
                                   #返回：被删除的DataFrame

#车辆的特征中有数值型和类别性，数值型的特征注意进行范围标准化，类别型的特征转化为one-hot encoding的形式
X=data_train.drop(['Score','Id'], axis=1)                
X=pd.get_dummies(X)                                                   #用pd.get_dummies方法进行one-hot编码                                  
X=X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))       #min-max标准化（Min-Max Normalization）
                                                                      #也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]

test_X=data_test.drop(['Id'], axis=1)                
test_X=pd.get_dummies(test_X)
test_X=test_X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))        #min-max标准化（Min-Max Normalization）
                                                                     #也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]之间

X=np.array(X)
X=pd.DataFrame(X)                           #加这两行是为了让dataframe的列名变为按序排列的数字，为什么用原来的列名总会出现使得X^T*X为奇异矩阵
test_X=np.array(test_X)
test_X=pd.DataFrame(test_X)                 
X_total=pd.concat([X,test_X])               #将训练测试集合并起来，统一进行数据处理
cols=[x for i,x in enumerate(X_total.columns) if X_total.iat[0,i]==0]              #得到全为0的列
X_total=X_total.drop(cols,axis=1)                                                  #删掉全为0的列
X=X_total[0:32000]
test_X=X_total[32000:40000]

# 将数据分为训练集和校验集
train_X, valid_X, train_Y, valid_Y = train_test_split(X.values, Y.values, test_size=0.25)

#多元线性回归
#梯度下降法    
def multiplelr(dataset_x,dataset_y):
    
    #参数初始化   
    alpha=0.01                          #步长初始化
    theta=1e-100                           #阈值初始化
    r,c=x.shape
    num_iter=r                     #最大迭代次数
    B=np.random.random([c,1])
    
    #计算梯度方向delta_B
    delta_B=(-dataset_x.T.dot(dataset_y.reshape(r,1))+dataset_x.T.dot(dataset_x).dot(B.T))/r
    
    #计算新的B值
    B=B-alpha*delta_B.T 
    
    #计算损失值
    loss=np.zeros(r)
    J=np.zeros(r)
    for l in range(1,num_iter):
        J[l]=1/r*(dataset_y[l]-dataset_x[l,:].dot(B.T)-loss[l-1])**2           #计算本次目标函数
        loss[l]=1/r*np.linalg.norm(J[l]-J[l-1])**2                     #计算新的损失值
        
        if loss[l]<theta:                                              #判断损失函数是否小于阈值
            print('迭代次数为：',l)
            print(loss[l])
            break
        else:
            delta_B=(-dataset_x.T.dot(dataset_y.reshape(r,1))+dataset_x.T.dot(dataset_x).dot(B.T))/r
            B=B-alpha*delta_B.T 
    
    return B
   
#线性回归模型的构建
B_new=multiplelr(train_X,train_Y)
Y_pred=valid_X.dot(B_new.T)

#输出校验集的均方根误差
rms = np.sqrt(mean_squared_error(valid_Y, Y_pred))
print(rms)

#用刚训练的模型对test.csv中的数据进行测试
Score=test_X.dot(B_new.T)
Score=np.array(Score)
Score=Score.reshape(Score.shape[0])       #输出测试集的分数

#将Id,Score存入csv文件
Id=data_test.Id
dataset=list(zip(Id,Score))
df=pd.DataFrame(data=dataset,columns=('Id','Score'))
df.to_csv('submission.csv',index=False)         #将数据写入csv文件中，不需要索引列


