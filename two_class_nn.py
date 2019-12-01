import pandas
from pylab import *
import keras as ks
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from keras.optimizers import Adam,SGD,sgd
from keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
# ...

# save loss and acc
class LossHistory(ks.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = [i for i in range(len(self.losses[loss_type]))]
        iindex = np.arange(0, len(self.losses[loss_type]), len(self.losses[loss_type]) / 200)
        iindex = list(iindex)
        plt.figure()
        # acc
        plt.plot(iters[iindex], np.array(self.accuracy[loss_type])[iindex], 'r', label='train acc')
        # loss
        plt.plot(iters[iindex], np.array(self.losses[loss_type])[iindex], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters[iindex], np.array(self.val_acc[loss_type])[iindex], 'b', label='val acc')
            # val_loss
            plt.plot(iters[iindex], np.array(self.val_loss[loss_type])[iindex], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        ylim(0, 1.5)
        plt.legend(loc="upper right")
        plt.savefig('fig.png')

        plt.show()

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
kdd_data_10percent = pandas.read_csv("kddcup.data_10_percent_corrected.csv", header=None, names = col_names)
kdd_data_test = pandas.read_csv("corrected.csv", header=None, names = col_names)
kdd_data_10percent.describe()


features = kdd_data_10percent[col_names[:-1]].astype(float)
np_features = np.array(features)
test_features = kdd_data_test[col_names[:-1]].astype(float)
test_np_features = np.array(test_features)

# labels
labels = kdd_data_10percent['label'].copy()
np_labels = np.array(labels)
np_labels = np.array(np_labels, dtype = np.float)

test_labels = kdd_data_test['label'].copy()
test_np_labels = np.array(test_labels)
test_np_labels = np.array(test_np_labels, dtype = np.float)


# Feature scaling
np_features = StandardScaler().fit_transform(np_features)
test_np_features = StandardScaler().fit_transform(test_np_features)

np_1000_features = np_features
np_1000_labels = np_labels

# data for Testing
np_feature_test = test_np_features
np_labels_test = test_np_labels

labels_for_nn = []
np_labels_test_nn = []


for i in range(len(np_1000_labels)):
    tmp_label = [0] * 2
    tmp_label[int(np_1000_labels[i])] = 1
    labels_for_nn.append(tmp_label)

for i in range(len(np_feature_test)):
    tmp_label = [0] * 2
    tmp_label[int(np_labels_test[i])] = 1
    np_labels_test_nn.append(tmp_label)

train_data_num = int(len(np_1000_features) * 0.8)


model = ks.models.Sequential()
model.add(Dense(16, input_dim=41))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
history = LossHistory()

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
# model.fit(x=data_train,y=labels_train,batch_size=128,nb_epoch=5000,verbose=1,validation_data=(data_test,labels_test),callbacks=[history])
model.fit(x=np_1000_features[:train_data_num],y=labels_for_nn[:train_data_num],batch_size=100,nb_epoch=2,verbose=1,validation_data=(np_1000_features[train_data_num:],labels_for_nn[train_data_num:]), callbacks=[history])
# history.loss_plot('epoch')

predicts = model.predict(np_feature_test)
np_labels_test_nn = np.array(np_labels_test_nn)
# Y_pred = [np.argmax(y) for y in predicts]
# Y_valid = [np.argmax(y) for y in np_labels_test_nn]

# precision = precision_score(Y_valid, Y_pred, average='weighted')
# recall = recall_score(Y_valid, Y_pred, average='weighted')
# f1_score = f1_score(Y_valid, Y_pred, average='weighted')
# accuracy_score = accuracy_score(Y_valid, Y_pred)
# precision = precision_score(np_labels_test_nn[:, 1], predicts[:, 1], average='weighted')
# recall = recall_score(np_labels_test_nn[:, 1], predicts[:, 1], average='weighted')
# f1_score = f1_score(np_labels_test_nn[:, 1], predicts[:, 1], average='weighted')
# accuracy_score = accuracy_score(np_labels_test_nn[:, 1], predicts[:, 1])
# print("Precision_score:",precision)
# print("Recall_score:",recall)
# print("F1_score:",f1_score)
# print("Accuracy_score:",accuracy_score)

fpr, tpr, thresholds_keras = roc_curve(np_labels_test_nn[:, 1], predicts[:, 1])
auc = auc(fpr, tpr)
print("AUC : ", auc)


plt.figure()
plt.plot(fpr, tpr, label='DNN Model  area = {:.3f})'.format(auc))

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('all_Log_ROC_DNN')
plt.show()
