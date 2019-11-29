import pandas as pd
import numpy as np
from time import time
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler  # install scipy package
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn.preprocessing import label_binarize
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

class IntrusionDetector:

    def __init__(self, train_data_path, test_kdd_path):
        self.train_kdd_path = train_data_path
        self.test_kdd_path = test_kdd_path

        self.train_kdd_data = []
        self.test_kdd_data = []

        self.train_kdd_numeric = []
        self.test_kdd_numeric = []

        self.train_kdd_binary = []
        self.test_kdd_binary = []

        self.train_kdd_nominal = []
        self.test_kdd_nominal = []

        self.train_kdd_label_2classes = []
        self.test_kdd_label_2classes = []
        #read data from file
        self.get_data()


    def get_data(self):
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
        self.train_kdd_data = pd.read_csv(self.train_kdd_path, header=None, names = col_names)
        self.test_kdd_data = pd.read_csv(self.test_kdd_path, header=None, names = col_names)
        self.train_kdd_data.describe()

    # To reduce labels into "Normal" and "Abnormal"
    def get_2classes_labels(self):
        label_2class = self.train_kdd_data['label'].copy()
        self.train_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))
        self.train_kdd_label_2classes = label_binarize(self.train_kdd_label_2classes, classes=[0, 1, 2])

        label_2class = self.test_kdd_data['label'].copy()
        self.test_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))
        self.test_kdd_label_2classes = label_binarize( self.test_kdd_label_2classes, classes=[0, 1, 2])

    def preprocessor(self):
        # prepare 2 classes label for "abnormal" and "normal"
        self.get_2classes_labels()

        nominal_features = ["protocol_type", "service", "flag"]  # [1, 2, 3]
        binary_features = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login",]  # [6, 11, 13, 14, 20, 21]
        numeric_features = [
            "duration", "src_bytes",
            "dst_bytes", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "num_compromised", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        #convert nominal features to numeric features
        #nominal features: ["protocol_type", "service", "flag"]
        self.train_kdd_nominal = self.train_kdd_data[nominal_features].astype(float)
        self.test_kdd_nominal = self.test_kdd_data[nominal_features].astype(float)
        # normalize
        # self.train_kdd_nominal = StandardScaler().fit_transform(self.train_kdd_nominal)
        # self.test_kdd_nominal = StandardScaler().fit_transform(self.test_kdd_nominal)


        self.train_kdd_binary = self.train_kdd_data[binary_features].astype(float)
        self.test_kdd_binary = self.test_kdd_data[binary_features].astype(float)
        # normalize
        # self.train_kdd_binary = StandardScaler().fit_transform(self.train_kdd_binary)
        # self.test_kdd_binary = StandardScaler().fit_transform(self.test_kdd_binary)

        # Standardizing and scaling numeric features
        self.train_kdd_numeric = self.train_kdd_data[numeric_features].astype(float)
        self.test_kdd_numeric = self.test_kdd_data[numeric_features].astype(float)
        # normalize
        self.train_kdd_numeric = StandardScaler().fit_transform(self.train_kdd_numeric)
        self.test_kdd_numeric = StandardScaler().fit_transform(self.test_kdd_numeric)

    def feature_reduction_ICA(self):
        pass
    def feature_reduction_PCA(self):

        numeric_pca = sklearnPCA(n_components=11)
        numeric_pca = numeric_pca.fit(self.train_kdd_numeric)
        # numeric_pca = numeric_pca.fit(np.concatenate((self.train_kdd_numeric, self.test_kdd_numeric), axis=0))
        self.train_kdd_numeric = numeric_pca.transform(self.train_kdd_numeric)
        self.test_kdd_numeric = numeric_pca.transform(self.test_kdd_numeric)
        # self.train_kdd_numeric = numeric_pca.fit_transform(self.train_kdd_numeric)
        # self.test_kdd_numeric = numeric_pca.fit_transform(self.test_kdd_numeric)

        binary_features_pca = sklearnPCA(n_components=5)
        # binary_features_pca = binary_features_pca.fit(np.concatenate((self.train_kdd_binary, self.test_kdd_binary), axis=0))
        # self.train_kdd_binary = binary_features_pca.transform(self.train_kdd_binary)
        # self.test_kdd_binary = binary_features_pca.transform(self.test_kdd_binary)
        self.train_kdd_binary = binary_features_pca.fit_transform(self.train_kdd_binary)
        self.test_kdd_binary = binary_features_pca.fit_transform(self.test_kdd_binary)

        nominal_features_pca = sklearnPCA(n_components=2)
        self.train_kdd_nominal = nominal_features_pca.fit_transform(self.train_kdd_nominal)
        self.test_kdd_nominal = nominal_features_pca.fit_transform(self.test_kdd_nominal)

    def format_data(self):

        kdd_train_data = np.concatenate([self.train_kdd_numeric, self.train_kdd_binary, self.train_kdd_nominal], axis=1)
        kdd_test_data = np.concatenate([self.test_kdd_numeric, self.test_kdd_binary, self.test_kdd_nominal], axis=1)
        #
        # kdd_train_data = np.concatenate([kdd_train_data, self.train_kdd_label_2classes],axis=1)
        # # kdd_test_data = np.concatenate([self.test_kdd_numeric, self.test_kdd_binary, self.test_kdd_nominal, self.test_kdd_label_2classes], axis=1)
        # kdd_test_data = np.concatenate([kdd_test_data, self.test_kdd_label_2classes], axis=1)
        # self.X_train, self.X_test, self.y_train, self.y_test = kdd_train_data[:, :-1], kdd_test_data[:, :-1], kdd_train_data[:-1], kdd_test_data[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = kdd_train_data, kdd_test_data, self.train_kdd_label_2classes, self.test_kdd_label_2classes

    def predicting(self, model, model_name):
        # Predict
        predicts = model.predict(self.X_test)
        print("Classifier:")
        accuracy = accuracy_score(self.y_test, predicts)
        print("Accuracy: ", accuracy)

        fpr1_gnb = {}
        tpr1_gnb = {}
        roc_dict = [0, 0, 0]
        predicts_proba = model.predict_proba(self.X_test)

        for i in range(3):
            # roc_dict[i] = roc_auc_score(self.y_test[:, i], predicts[:, i])
            # model_roc_auc = roc_auc_score(self.y_test[:, i], predicts[:, i])
            # fpr1_gnb[i], tpr1_gnb[i], _ = roc_curve(self.y_test[:, i], model.decision_function(self.X_test)[:, i])
            fpr1_gnb[i], tpr1_gnb[i], _ = roc_curve(self.y_test[:, i], predicts_proba[:, i])
            roc_dict[i] = auc(fpr1_gnb[i], tpr1_gnb[i])
            print("Auc: ", roc_dict[i])

            con_matrix = confusion_matrix(self.y_test[:, i], predicts[:, i], labels=[0, 1])
            # con_matrix = confusion_matrix(y_test, predicts, labels=["normal.", "abnormal."])
            print("confusion matrix:")
            print(con_matrix)
            precision = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
            recall = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            tpr = recall
            fpr = con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1])
            print("Precision:", precision)
            print("Recall:", recall)
            print("TPR:", tpr)
            print("FPR:", fpr)
        # scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        # print('Cross validation Score:', scores)

        plt.figure()
        colors = ['aqua', 'darkorange', 'cornflowerblue']
        classes = ['normal', 'smurf', 'attack']
        for i in range(3):
            plt.plot(fpr1_gnb[i], tpr1_gnb[i], color=colors[i], lw=2, label='%s Model of %s (area = %0.2f)' %(model_name, classes[i], roc_dict[i]) )
        
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('3_Log_ROC_%s' %model_name)
        plt.show()
    def boost_predicting(self, models, model_name):
        # Predict
        plt.figure()
        for i in range(3):
            predicts = models[i].predict(self.X_test[:50000])
            print("Classifier:")
            accuracy = accuracy_score(self.y_test[:50000], predicts)
            print("Accuracy: ", accuracy)

            model_roc_auc = roc_auc_score(self.y_test[:50000], predicts)
            print("Auc: ", model_roc_auc)
            fpr1_gnb, tpr1_gnb, thresholds1_gnb = roc_curve(self.y_test[:50000], models[i].predict_proba(self.X_test[:50000])[:, 1])

            con_matrix = confusion_matrix(self.y_test[:50000], predicts, labels=[0, 1])
            # con_matrix = confusion_matrix(y_test, predicts, labels=["normal.", "abnormal."])
            print("confusion matrix:")
            print(con_matrix)
            precision = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
            recall = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[0][1])
            tpr = recall
            fpr = con_matrix[1][0] / (con_matrix[1][0] + con_matrix[1][1])
            print("Precision:", precision)
            print("Recall:", recall)
            print("TPR:", tpr)
            print("FPR:", fpr)
        # scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        # print('Cross validation Score:', scores)

            plt.plot(fpr1_gnb, tpr1_gnb, label='%s Model  (area = %0.2f)' %(model_name[i], model_roc_auc) )

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        # plt.savefig('Log_ROC_boost')
        # plt.savefig('Log_ROC_%s' %model_name)
        plt.show()
    def SGD_Classifier(self):

        # Create a model
        model = SGDClassifier(loss="log")
        model.fit(self.X_train, self.y_train)

        # Predict
        self.predicting(model, "SGD")
    def bayes_classifier(self):
        model = OneVsRestClassifier(GaussianNB())
        print(self.y_train.shape)
        model.fit(self.X_train, self.y_train)

        # Predict
        self.predicting(model, "CNB")


    def knn_classifier(self):

        #Load classifier from Pickle
        # model=pickle.load(open("knearestneighbor.pickle", "rb"))
        model = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=5))
        model.fit(self.X_train, self.y_train)
        with open('knearestneighbor.pickle','wb') as f:
            pickle.dump(model,f)
        print('model trained')

        # predict
        self.predicting(model, "KNN")

    def svm_classifier(self):
        # Create SVM classification object
        model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=0.1, decision_function_shape='ovr', verbose= True, probability=True))
        # model = svm.SVC(kernel='rbf', C=0.8, gamma=20, decision_function_shape='ovr', probability=True)
        model.fit(self.X_train[:50000], self.y_train[:50000])

        # Predict Output
        self.predicting(model, 'SVM')


    def decision_tree_classifier(self):
        model = OneVsRestClassifier(tree.DecisionTreeClassifier(criterion="entropy"))
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "ID3_DTC")
    def random_forest_classifier(self):
        model = OneVsRestClassifier(ensemble.RandomForestClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "RFC")
    def adaboost_classifier(self):
        model = OneVsRestClassifier(ensemble.AdaBoostClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "AdaBoost")
    def bagging_classifier(self):
        model = OneVsRestClassifier(ensemble.BaggingClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "bagging")
    def XGBoost(self):
        model = OneVsRestClassifier(XGBClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "XGBoost")
    def gradient_boosting_classifier(self):
        model = OneVsRestClassifier(ensemble.GradientBoostingClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "grad_boost")
    def xgb_rf_classifier(self):
        model = OneVsRestClassifier(XGBRFClassifier())
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "XGBRF")
    def Boost(self):
        model1 = OneVsRestClassifier(XGBClassifier())
        model1.fit(self.X_train, self.y_train)
        model2 = OneVsRestClassifier(ensemble.GradientBoostingClassifier())
        model2.fit(self.X_train, self.y_train)
        model3 = OneVsRestClassifier(XGBRFClassifier())
        model3.fit(self.X_train, self.y_train)

        self.boost_predicting([model1, model2, model3], ["XGBoost", "grad_boost", "XGBRF"])
def main():
    # Data path
    cwd = os.getcwd()  # current directory path
    kdd_data_path_train = cwd + "/three_kddcup.data_10_percent_corrected.csv"
    kdd_data_path_test = cwd + "/three_corrected.csv"

    i_detector = IntrusionDetector(kdd_data_path_train, kdd_data_path_test)
    i_detector.preprocessor()
    i_detector.feature_reduction_PCA()
    i_detector.format_data()

    while (True):
        print("\n\n")
        print("0. SGD Classifier")
        print("1. Naive Bayes Classifier")
        print("2. SVM Classifier")
        print("3. KNN Classifier")
        print("4. Decision Tree Classifier")
        print("5. Random Forest Classifier")
        print("6. AdaBoost Classifier")
        print("7. Bagging Classifier")
        print("8. XGBoost Classifier")
        print("9. Gradient Boosting Classifier")
        print("10. XGB RF Classifier")
        print("11. Three Classifier")
        print("12. Quit")
        option = input("Please enter a value:")
        if option == "0":
            i_detector.SGD_Classifier()
        elif option == "1":
            # for j in range(1, 39):
            #     print("============================================")
            #     print("Now the %d-th features" %j)
            #     i_detector.format_data(j)
            #     i_detector.bayes_classifier()
            #     print("============================================")
            i_detector.bayes_classifier()
        elif option == "2":
            import time
            time1 = time.time()
            i_detector.svm_classifier()
            print(time.time() - time1)
        elif option == "3":
            i_detector.knn_classifier()
        elif option == "4":
            i_detector.decision_tree_classifier()
        elif option == "5":
            i_detector.random_forest_classifier()
        elif option == "6":
            i_detector.adaboost_classifier()
        elif option == "7":
            i_detector.bagging_classifier()
        elif option == "8":
            i_detector.XGBoost()
        elif option == "9":
            i_detector.gradient_boosting_classifier()
        elif option == "10":
            i_detector.xgb_rf_classifier()
        elif option == "11":
            i_detector.Boost()

        elif option == "12":
            break


if __name__ == '__main__':
    main()
