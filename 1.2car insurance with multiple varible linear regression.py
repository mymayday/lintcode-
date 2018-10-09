import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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
X=pd.get_dummies(X) 
X=X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))       #min-max标准化（Min-Max Normalization）
                                                                     #也称为离差标准化，是对原始数据的线性变换，使结果值映射到[0 - 1]

#检验数据
print(X.describe())#数据描述，会显示最值，平均数等信息，可以简单判断数据中是否有异常值
print(X[X.isnull()==True].count())#检验缺失值，若输出为0，说明该列没有缺失值

# 将数据分为训练集和校验集
train_X, valid_X, train_Y, valid_Y = train_test_split(X.values, Y.values, test_size=0.25)

#建立线性回归模型
model = LinearRegression() 
model.fit(train_X,train_Y)
a  = model.intercept_        #截距 
b = model.coef_              #回归系数 
print("最佳拟合线:截距",a,",回归系数：",b)

score = model.score(valid_X,valid_Y )
#输出校验集的预测值
Y_pred = model.predict(valid_X)

#输出校验集的均方根误差
rms = np.sqrt(mean_squared_error(valid_Y, Y_pred))
print(rms)

#用刚训练的模型对test.csv中的数据进行测试
test_X=data_test.drop(['Id'], axis=1)                
test_X=pd.get_dummies(test_X)
test_X=test_X.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))) 

#输出测试集的分数
Score = model.predict(test_X)

#将Id,Score存入csv文件
Id=data_test.Id
dataset=list(zip(Id,Score))
df=pd.DataFrame(data=dataset,columns=('Id','Score'))

df.to_csv('submission.csv',index=False)          #index=False，根据要求索引列不需要存入
