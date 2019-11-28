# coding: utf-8
import tkinter as tk
import tkinter.filedialog
import threading
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import time, os
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score

##　界面初始化
window = tk.Tk()
window.title('SVM-KDD99')
window.geometry('700x600')

data_path = None


def selectClick():
    global data_path
    fileName = tkinter.filedialog.askopenfilename()
    data_path = fileName
    fileVar.set(fileName)


def trainClick():
    th = threading.Thread(target=trainThread, args=(data_path,))
    th.setDaemon(True)  # 守护线程
    th.start()


def valClick():
    th = threading.Thread(target=valThread)
    th.setDaemon(True)  # 守护线程
    th.start()


fileVar = tk.StringVar()
fileVar.set(u'未选择')
accVar = tk.StringVar()
msgVar = tk.StringVar()
predVar = tk.StringVar()
realVar = tk.StringVar()

dataLabel = tk.Label(window, text=u'训练数据', font=('Arial', 12))
datapathLabel = tk.Label(window, bg='white', textvariable=fileVar, width=50, font=('Arial', 12))
selectButton = tk.Button(window, text='选择训练数据', fg='green', width=10, command=selectClick)

trainButton = tk.Button(window, text='开始训练', fg='red', width=10, command=trainClick)
valaccLabel = tk.Label(window, text=u'验证集精度：', font=('Arial', 12))
valaccDisp = tk.Label(window, bg='white', textvariable=accVar, width=10, font=('Arial', 12))

valButton = tk.Button(window, text='测试', fg='blue', width=10, command=valClick)
predLabel = tk.Label(window, text=u'预测攻击类型：', font=('Arial', 12))
predDisp = tk.Label(window, textvariable=predVar, width=15, bg='white', font=('Arial', 12))
realLabel = tk.Label(window, text=u'实际攻击类型：', font=('Arial', 12))
realDisp = tk.Label(window, textvariable=realVar, bg='white', width=15, font=('Arial', 12))

statusLabel = tk.Label(window, bg='white', fg='red', textvariable=msgVar, font=('Arial', 12), width=75, height=22,
                       wraplength=580, justify='left', anchor='nw')

dataLabel.place(x=10, y=10)
datapathLabel.place(x=80, y=10)
selectButton.place(x=550, y=10)

trainButton.place(x=10, y=50)
valaccLabel.place(x=100, y=50)
valaccDisp.place(x=200, y=50)

valButton.place(x=10, y=100)
predLabel.place(x=100, y=100)
predDisp.place(x=210, y=100)
realLabel.place(x=360, y=100)
realDisp.place(x=480, y=100)

statusLabel.place(x=10, y=140)


def valThread():
    # #### 加载数据
    #  msgVar.set("./test_data.csv\n" )
    #  df = pd.read_csv("./test_data.csv")
    msgVar.set(msgVar.get() + u"读取测试数据...\n")
    X_val = np.load('X_test.npy')
    y_val = pd.Series(np.load('y_test.npy'))

    msgVar.set(msgVar.get() + u"加载模型...\n")
    model = joblib.load("model.m")

    msgVar.set(msgVar.get() + u"随机选择一个测试样本进行测试...\n")
    score = cross_val_score(model, X_val, y_val)
    print("Cross val score is", score)

    predict = model.predict(X_val)
    accuracy = accuracy_score(y_val, predict)
    print("Accuracy:", accuracy)

    model_roc_auc = roc_auc_score(y_val, predict)
    print(model_roc_auc)

    sel = np.random.choice(range(len(X_val)))
    y_real = y_val.iloc[sel]
    y_pred = model.predict(X_val[sel].reshape(1, -1))
    print(type(y_pred[0]))
    predVar.set((y_pred[0]))
    realVar.set(y_real)


def trainThread(data_path=data_path):
    # #### 加载数据
    if data_path == None or not os.path.isfile(data_path):
        msgVar.set(u"请选择正确的数据集\n")
        return
    msgVar.set("%s\n" % data_path)
    df = pd.read_csv(data_path, engine='python')
    msgVar.set(msgVar.get() + u"读取数据...\n")
    print(df.shape)

    # #### stratified sampling
    msgVar.set(msgVar.get() + u"解析数据...\n")
    X = df.copy().drop(['target'], axis=1)
    Y = df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, Y, stratify=Y, test_size=0.2)

    msgVar.set(msgVar.get() + "size for y_train: %s\n" % y_train.shape)
    msgVar.set(msgVar.get() + "size of x_train: %s x %s\n" % X_train.shape)
    msgVar.set(msgVar.get() + "size for y_test: %s\n" % y_val.shape)
    msgVar.set(msgVar.get() + "size of x_test: %s x %s\n" % X_val.shape)
    print("size for y_train: %s" % y_train.shape)
    print("size of x_train: %s x %s" % X_train.shape)
    print("size for y_test: %s" % y_val.shape)
    print("size of x_test: %s x %s" % X_val.shape)

    ##离散的特征
    print(X_train.iloc[:, [1, 2, 3]].head())

    # #### 特征处理
    # ##### Onehot
    X_train_onthot = pd.get_dummies(X_train, columns=["protocol_type", "service", "flag"])
    X_val_onehot = pd.get_dummies(X_val, columns=["protocol_type", "service", "flag"])
    missing_cols = set(X_train_onthot.columns) - set(X_val_onehot.columns)
    print('missing feature in x_val_onehot:')

    for c in missing_cols:
        print(c)
        X_val_onehot[c] = 0
    X_val_onehot = X_val_onehot[X_train_onthot.columns]

    msgVar.set(msgVar.get() + "size of x_train_onehot: %s x %s\n" % X_train_onthot.shape)
    msgVar.set(msgVar.get() + "size of x_val_onehot: %s x %s\n" % X_val_onehot.shape)
    print("size of x_train_onehot: %s x %s" % X_train_onthot.shape)
    print("size of x_val_onehot: %s x %s" % X_val_onehot.shape)

    # ##### MinMax Processing

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_onthot)
    X_train_onthot = scaling.transform(X_train_onthot)
    X_val_onehot = scaling.transform(X_val_onehot)

    msgVar.set(msgVar.get() + u"正在训练，请耐心等待...\n")
    model = svm.SVC(kernel='linear', C=1, verbose=True, decision_function_shape="ovo", max_iter=-1)
    model.fit(X_train_onthot, y_train)
    val_acc = model.score(X_val_onehot, y_val)  ##输出测试准确率
    print('val_acc:%.4f' % val_acc)
    accVar.set("%.4f%%" % (val_acc % 100))
    msgVar.set(msgVar.get() + u"训练完成\n")

    np.save('X_test.npy', X_val_onehot)  ##保存下来，用于测试
    np.save('y_test.npy', y_val)

    msgVar.set(msgVar.get() + u"保存模型...\n")
    joblib.dump(model, "model.m")


window.mainloop()