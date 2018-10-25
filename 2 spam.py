import pandas as pd                                                #导入pandas包用于数据集的读取
import numpy as np                                                 #导入numpy用于矩阵计算
import nltk
from nltk.tokenize import word_tokenize       
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split              # train_test_split函数将数据集中的数据按一定比例划分为训练集和校验集
import re                                                          #导入正则表达式模块


#数据的导入
data_train=pd.read_csv('train.csv',names=['Label', 'Text'])                      #读取训练数据
data_test=pd.read_csv('test.csv',names=['Id', 'Text'])                           #读取测试数据
#加上names可以解决读进来错乱的问题
#读出来的数据会多第一行Label、Text，用header=None也没办法解决，只能下来再删掉第一行
data_train=data_train.drop([0])
data_test=data_test.drop([0])

X_total=pd.concat([data_train,data_test])               #将训练测试集合并起来，统一进行数据处理

#清洗数据
def clean_text(message_text): 
    message_list=[]                     
    s = nltk.stem.snowball.EnglishStemmer()        # 词干提取

    for text in message_text: 
        text = re.sub('���', "'", text)
        text = re.sub('��', "'", text)
        text = re.sub('�', "'", text)
        text = text.lower() # 小写转换
        # 删除非字母、数字字符 
        text = re.sub(r"[^a-z']", " ", text) 
        # 恢复常见的简写 
        text = re.sub(r"what's", "what is ", text) 
        text = re.sub(r"\'s", " ", text) 
        text = re.sub(r"\'ve", " have ", text) 
        text = re.sub(r"can't", "can not ", text) 
        text = re.sub(r"cannot", "can not ", text) 
        text = re.sub(r"n't", " not ", text) 
        text = re.sub(r"\'m", " am ", text) 
        text = re.sub(r"\'re", " are ", text) 
        text = re.sub(r"\'d", " will ", text) 
        text = re.sub(r"ain\'t", " are not ", text) 
        text = re.sub(r"aren't", " are not ", text) 
        text = re.sub(r"couldn\'t", " can not ", text) 
        text = re.sub(r"didn't", " do not ", text) 
        text = re.sub(r"doesn't", " do not ", text) 
        text = re.sub(r"don't", " do not ", text) 
        text = re.sub(r"hadn't", " have not ", text) 
        text = re.sub(r"hasn't", " have not ", text) 
        text = re.sub(r"\'ll", " will ", text)
                     
        new_text = ""
        for word in word_tokenize(text):
            new_text = new_text + " " + s.stem(word)
        
        message_list.append(new_text)
             
    return message_list
    
Total_Text=clean_text(X_total.Text)

x_total=np.array(Total_Text)
y=data_train.Label.map({'ham':0, 'spam':1})  

# 数据的TF-IDF信息计算
total_text_list = list(Total_Text)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(total_text_list)
x_total = text_vector.transform(Total_Text)
x_total=x_total.toarray()

x=x_total[0:5572]
test_X=x_total[5572:6687]

def NaiveBayes(x,y):
    
    x=x.astype('float64')
    N,M=x.shape                     #N为短信数，M为所有短信中出现的单词数
    P_prior=sum(y)/N                #计算训练集中短信为垃圾短信的 先验概率
        
    P_0_total=1                      #用P_0_total记录每条非垃圾短信的总词汇数，初始化为1，避免分母为0
    P_1_total=1                      #用P_1_total记录每条垃圾短信的总词汇数
    
    #将所有词的出现次数初始化为1
    P_0_count=np.ones(M)
    P_1_count=np.ones(M)
    
    for i in range(N):
        if y[i]==1:                                     #y[i]=1时为垃圾短信，记录短信中单词的词频
            P_1_count+=np.array(x[i])
            P_1_total+=sum(x[i])
        else:
            P_0_count+=np.array(x[i])                   #y[i]=0时为非垃圾短信，记录短信中单词的词频
            P_0_total+=sum(x[i])
            
    #计算likelihood
    P_0_condition=P_0_count/P_0_total
    P_1_condition=P_1_count/P_1_total
    #为保证概率不会过小，取对数
    P_0_condition=np.log(P_0_condition)
    P_1_condition=np.log(P_1_condition)
    
    return P_prior,P_0_condition,P_1_condition     
    
    
# 贝叶斯分类器
def predict(x,P_prior,P_0_condition,P_1_condition):
    pred = []
    for item in x:
        P_0=sum(item*P_0_condition)+np.log(1-P_prior)
        P_1=sum(item*P_1_condition)+np.log(P_prior)
        if P_1> P_0:
            pred.append(1)
        else:
            pred.append(0)
    return np.array(pred)
    
# 将数据分为训练集和校验集
train_X, valid_X, train_Y, valid_Y = train_test_split(x, y.values, test_size=0.25)
P_p,P_0_c,P_1_c=NaiveBayes(train_X,train_Y)
Y_pred=predict(valid_X,P_p,P_0_c,P_1_c)

from sklearn.metrics import accuracy_score
print('Accuracy score: ', format(accuracy_score(valid_Y, Y_pred)))         #查看校验集的准确率

Y_test=predict(test_X,P_p,P_0_c,P_1_c)                                     #对测试集进行分类

data_test['Label']=''
data_test.drop(['Text'],axis=1,inplace=True)

for index in range(len(data_test.Id)):
    if Y_test[index]==1:
        data_test.loc[index,'Label']='spam'
    else:
        data_test.loc[index,'Label']='ham'
        
Id=data_test.Id
dataset=list(zip(Id,data_test.Label))
df=pd.DataFrame(data=dataset,columns=('SmsId','Label'))
#将数据写入csv文件中，不需要索引列
df.to_csv('submission.csv',index=False)

