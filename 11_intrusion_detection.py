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
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from xgboost import XGBClassifier, XGBRFClassifier

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
        label_2class[label_2class != 'normal.'] = 'abnormal.'
        self.train_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))

        label_2class = self.test_kdd_data['label'].copy()
        label_2class[label_2class != 'normal.'] = 'abnormal.'
        self.test_kdd_label_2classes = label_2class.values.reshape((label_2class.shape[0], 1))

    def preprocessor(self):
        # prepare 2 classes label for "abnormal" and "normal"
        self.get_2classes_labels()

        # nominal_features = ["protocol_type", "service", "flag"]  # [1, 2, 3]
        # binary_features = ["land", "num_failed_logins", "num_compromised",\
        #                    "root_shell", "num_outbound_cmds", "is_host_login"]  # [6, 11, 13, 14, 20, 21]
        # numeric_features = [
        #     "duration", "src_bytes",
        #     "dst_bytes", "wrong_fragment", "urgent", "hot",
        #     "logged_in", "su_attempted", "num_root",
        #     "num_file_creations", "num_shells", "num_access_files",
        #     "is_guest_login", "count", "srv_count", "serror_rate",
        #     "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        #     "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        #     "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        #     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        #     "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        # ]
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
        self.train_kdd_nominal = self.train_kdd_data[nominal_features].stack().astype('category').unstack()
        self.test_kdd_nominal = self.test_kdd_data[nominal_features].stack().astype('category').unstack()

        self.train_kdd_nominal = np.column_stack((self.train_kdd_nominal["protocol_type"].cat.codes, \
                                               self.train_kdd_nominal["service"].cat.codes, \
                                               self.train_kdd_nominal["flag"].cat.codes))
        self.test_kdd_nominal = np.column_stack((self.test_kdd_nominal["protocol_type"].cat.codes, \
                                               self.test_kdd_nominal["service"].cat.codes, \
                                               self.test_kdd_nominal["flag"].cat.codes))
        #print (kdd_nominal_encoded)

        self.train_kdd_binary = self.train_kdd_data[binary_features]
        self.test_kdd_binary = self.test_kdd_data[binary_features]

        # Standardizing and scaling numeric features
        self.train_kdd_numeric = self.train_kdd_data[numeric_features].astype(float)
        self.test_kdd_numeric = self.test_kdd_data[numeric_features].astype(float)
        # self.train_kdd_numeric = StandardScaler().fit_transform(self.train_kdd_numeric)
        # self.test_kdd_numeric = StandardScaler().fit_transform(self.test_kdd_numeric)

    def feature_reduction_ICA(self):
        pass
    def feature_reduction_PCA(self):
        # #compute Eigenvectors and Eigenvalues
        # mean_vec = np.mean(self.kdd_numeric, axis=0)
        # cov_mat = np.cov((self.kdd_numeric.T))
        #
        # # Correlation matrix
        # cor_mat = np.corrcoef((self.kdd_numeric.T))
        # eig_vals, eig_vecs = np.linalg.eig(cor_mat)
        #
        # # To check that the length of eig_vectors is 1
        # for ev in eig_vecs:
        #     np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
        # #print ('eigen_vector length is 1')
        #
        # #to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors and ignore the rest
        # # Make a list of (eigenvalue, eigenvector) tuples
        # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        #
        # # Sort and print the (eigenvalue, eigenvector) tuples from high to low
        # eig_pairs.sort()
        # eig_pairs.reverse()
        # #for i in eig_pairs:
        # #    print(i[0])
        #
        # #feature reduction
        # # just the 10 first items are greater 1 and one which is close to 1 => pick 11
        # matrix_w = np.hstack((eig_pairs[0][1].reshape(32,1),
        #                       eig_pairs[1][1].reshape(32,1),
        #                       eig_pairs[2][1].reshape(32,1),
        #                       eig_pairs[3][1].reshape(32,1),
        #                       eig_pairs[4][1].reshape(32,1),
        #                       eig_pairs[5][1].reshape(32,1),
        #                       eig_pairs[6][1].reshape(32,1),
        #                       eig_pairs[7][1].reshape(32,1),
        #                       eig_pairs[8][1].reshape(32,1),
        #                       eig_pairs[9][1].reshape(32,1),
        #                       eig_pairs[10][1].reshape(32,1)))
        # # projection to new feature space
        # self.kdd_numeric = self.kdd_numeric.dot(matrix_w)

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
        # pca_data = np.concatenate([kdd_train_data, kdd_test_data], axis=0)
        # ica_data = np.concatenate([kdd_train_data, kdd_test_data], axis=0)
        # # data_pca = sklearnPCA(n_components=13)
        # data_ica = FastICA(n_components=13)
        # # data_pca = data_pca.fit(pca_data)
        # data_ica = data_ica.fit(ica_data)
        # print(FastICA.get_params(data_ica))
        # kdd_train_data = data_ica.transform(kdd_train_data)
        # kdd_test_data = data_ica.transform(kdd_test_data)

        # kdd_train_data = np.concatenate([self.train_kdd_numeric, self.train_kdd_binary, self.train_kdd_nominal, self.train_kdd_label_2classes],axis=1)
        kdd_train_data = np.concatenate([kdd_train_data, self.train_kdd_label_2classes],axis=1)
        print(kdd_train_data.shape, self.train_kdd_label_2classes.shape)
        # kdd_test_data = np.concatenate([self.test_kdd_numeric, self.test_kdd_binary, self.test_kdd_nominal, self.test_kdd_label_2classes], axis=1)
        kdd_test_data = np.concatenate([kdd_test_data, self.test_kdd_label_2classes], axis=1)
        self.X_train, self.X_test, y_train, y_test = kdd_train_data[:, :-1], kdd_test_data[:, :-1], kdd_train_data[:,-1], kdd_test_data[:, -1]

        y_train[y_train == 'normal.'] = 0
        y_train[y_train == 'abnormal.'] = 1
        self.y_train = np.array(list(map(int, y_train)))
        y_test[y_test == 'normal.'] = 0
        y_test[y_test == 'abnormal.'] = 1
        self.y_test = np.array(list(map(np.int64, y_test)))
        # with open('data.csv', 'w') as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerows(self.X_train)

    def predicting(self, model, model_name):
        # Predict
        predicts = model.predict(self.X_test)
        print("Classifier:")
        accuracy = accuracy_score(self.y_test, predicts)
        print("Accuracy: ", accuracy)

        model_roc_auc = roc_auc_score(self.y_test, predicts)
        print("Auc: ", model_roc_auc)
        print(model.predict_proba(self.X_test).shape)

        fpr1_gnb, tpr1_gnb, thresholds1_gnb = roc_curve(self.y_test, model.predict_proba(self.X_test)[:, 1])

        con_matrix = confusion_matrix(self.y_test, predicts, labels=[0, 1])
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
        plt.plot(fpr1_gnb, tpr1_gnb, label='%s Model  (area = %0.2f)' %(model_name, model_roc_auc) )

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('Log_ROC_%s' %model_name)
        plt.show()
    def boost_predicting(self, models, model_name):
        # Predict
        plt.figure()
        for i in range(3):
            predicts = models[i].predict(self.X_test)
            print("Classifier:")
            accuracy = accuracy_score(self.y_test, predicts)
            print("Accuracy: ", accuracy)

            model_roc_auc = roc_auc_score(self.y_test, predicts)
            print("Auc: ", model_roc_auc)
            fpr1_gnb, tpr1_gnb, thresholds1_gnb = roc_curve(self.y_test, models[i].predict_proba(self.X_test)[:, 1])

            con_matrix = confusion_matrix(self.y_test, predicts, labels=[0, 1])
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
        plt.savefig('Log_ROC_boost')
        # plt.savefig('Log_ROC_%s' %model_name)
        plt.show()
    def SGD_Classifier(self):

        # Create a model
        model = SGDClassifier(loss="log")
        model.fit(self.X_train, self.y_train)

        # Predict
        self.predicting(model, "SGD")
    def bayes_classifier(self):
        # model_list = [GaussianNB, MultinomialNB, BaseDiscreteNB, BaseNB, BernoulliNB,ComplementNB]
        # Create a model
        model = GaussianNB()
        #Load classifier from Pickle
        # model=pickle.load(open("naivebayes.pickle", "rb"))
        # Train the model using the training sets

        model.fit(self.X_train, self.y_train)
        print(self.X_train.shape)
        print(self.y_train.shape)
        # with open('naivebayes.pickle','wb') as f:
        #     pickle.dump(model,f)

        # Predict
        self.predicting(model, "CNB")


    def knn_classifier(self):

        #Load classifier from Pickle
        # model=pickle.load(open("knearestneighbor.pickle", "rb"))
        model = neighbors.KNeighborsClassifier(n_neighbors=3)
        model.fit(self.X_train, self.y_train)
        with open('knearestneighbor.pickle','wb') as f:
            pickle.dump(model,f)
        print('model trained')

        # predict
        self.predicting(model, "KNN")

    def svm_classifier(self):
        # data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        # kdd_train_data, kdd_test_data = train_test_split(data, train_size=0.1)

        # Create SVM classification object
        model = svm.SVC(kernel='rbf', C=0.8, decision_function_shape='ovo', verbose= True, probability=True)
        # model = svm.SVC(kernel='rbf', C=0.8, gamma=20, decision_function_shape='ovr', probability=True)
        model.fit(self.X_train, self.y_train)

        # Predict Output
        self.predicting(model, 'SVM')


    def decision_tree_classifier(self):
        #data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        # data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)

        model = tree.DecisionTreeClassifier(criterion="entropy")
        print(tree.DecisionTreeClassifier.get_params(model))
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "DTC")
    def random_forest_classifier(self):
        #data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        # data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)

        model = ensemble.RandomForestClassifier()
        print(ensemble.RandomForestClassifier.get_params(model))
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "RFC")
    def adaboost_classifier(self):
        #data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        # data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)

        model = ensemble.AdaBoostClassifier()
        print(ensemble.AdaBoostClassifier.get_params(model))
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "AdaBoost")
    def bagging_classifier(self):
        #data = np.concatenate([self.kdd_numeric, self.kdd_label_2classes], axis=1)
        # data = np.concatenate([self.kdd_numeric, self.kdd_binary, self.kdd_nominal, self.kdd_label_2classes], axis=1)
        model = ensemble.BaggingClassifier()
        print(ensemble.BaggingClassifier.get_params(model))
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "bagging")
    def XGBoost(self):
        model = XGBClassifier()
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "XGBoost")
    def gradient_boosting_classifier(self):
        model = ensemble.GradientBoostingClassifier()
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "grad_boost")
    def xgb_rf_classifier(self):
        model = XGBRFClassifier()
        model.fit(self.X_train, self.y_train)
        self.predicting(model, "XGBRF")
    def Boost(self):
        model1 = XGBClassifier()
        model1.fit(self.X_train, self.y_train)
        model2 = ensemble.GradientBoostingClassifier()
        model2.fit(self.X_train, self.y_train)
        model3 = XGBRFClassifier()
        model3.fit(self.X_train, self.y_train)

        self.boost_predicting([model1, model2, model3], ["XGBoost", "grad_boost", "XGBRF"])
def main():
    # Data path
    cwd = os.getcwd()  # current directory path
    kdd_data_path_train = cwd + "/data/1-kddcup.data_10_percent_corrected"
    kdd_data_path_test = cwd + "/data/3-corrected.txt"

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
