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
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from keras.optimizers import Adam,SGD,sgd
from keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

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
        iters = range(len(self.losses[loss_type]))
        iindex = np.arange(0, len(self.losses[loss_type]), len(self.losses[loss_type]) / 200)
        iters = np.array(iters)
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
kdd_data_10percent = pandas.read_csv("three_kddcup.data_10_percent_corrected.csv", header=None, names = col_names)
kdd_data_test = pandas.read_csv("three_corrected.csv", header=None, names = col_names)
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
for i in range(41):
    d_min = min(np_features[:][i])
    d_max = max(np_features[:][i])
    np_features[:][i] -= d_min
    np_features[:][i] /= d_max
    test_d_min = min(test_np_features[:][i])
    test_d_max = max(test_np_features[:][i])
    test_np_features[:][i] -= test_d_min
    test_np_features[:][i] /= test_d_max


np_1000_features = np_features
np_1000_labels = np_labels

# data for Testing
np_feature_test = test_np_features
np_labels_test = test_np_labels

labels_for_nn = []
np_labels_test_nn = []


for i in range(len(np_1000_labels)):
    tmp_label = [0] * 3
    tmp_label[int(np_1000_labels[i])] = 1
    labels_for_nn.append(tmp_label)

for i in range(len(np_feature_test)):
    tmp_label = [0] * 3
    tmp_label[int(np_labels_test[i])] = 1
    np_labels_test_nn.append(tmp_label)

train_data_num = int(len(np_1000_features) * 0.9)


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
model.add(Dense(3))
model.add(Activation('softmax'))
history = LossHistory()

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
# model.fit(x=data_train,y=labels_train,batch_size=128,nb_epoch=5000,verbose=1,validation_data=(data_test,labels_test),callbacks=[history])
model.fit(x=np_1000_features[:train_data_num],y=labels_for_nn[:train_data_num],batch_size=100,nb_epoch=20,verbose=1,validation_data=(np_1000_features[train_data_num:],labels_for_nn[train_data_num:]), callbacks=[history])
history.loss_plot('epoch')

predicts = model.predict(np_feature_test)
np_labels_test_nn = np.array(np_labels_test_nn)

fpr, tpr, roc_dict = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(np_labels_test_nn[:, i], predicts[:, i])
    roc_dict[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
classes = ['normal', 'smurf', 'attack']
for i, color in zip(range(3), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,label='ROC curve of {0} (area = {1:0.2f})'.format(classes[i], roc_dict[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('3_Log_ROC_DNN')
plt.show()
