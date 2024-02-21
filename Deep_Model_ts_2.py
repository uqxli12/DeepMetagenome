# coding: utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, MaxPooling1D,AveragePooling1D, Flatten,Convolution1D,SpatialDropout1D, Dropout, GlobalAveragePooling1D,GlobalMaxPooling1D 
from keras.layers.normalization import BatchNormalization  
from keras.layers import LSTM 
from keras.layers.embeddings import Embedding
#in MacOs it can be 'from tensorflow.python.keras.layers.embeddings import Embedding'
from keras.callbacks import EarlyStopping
from keras.preprocessing import text, sequence
#For higher versions of keras, you need use "from keras_preprocessing import sequence" for using pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

from transformer import build_model
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import os
import seaborn as sns
from collections import Counter 
#Define MaxLen or MinLen first

MaxLen = 200
CutOffValue = 0.001
class Seq_Preprocess():
    def __init__(self,file):  
        #file is a file name (string) of the csv training database, containing two features of 'Sequence' and 'Classification' label
        self.file=file
    def file_reader(self):
        #Read a comma-separated values (csv) file into DataFrame
        model_f=pd.read_csv(self.file)  
        print(model_f.head())
        #drop duplicated
        model_f= model_f.drop_duplicates(subset=["Sequence"])
        #save the current dataset with duplicates removed
        model_f.to_csv("df_duplicateout.csv")  #a csv file naming "df_duplicateout.csv" will appear in the work directory
        #make the dataset reusable for next method 
        return model_f
    def plot_classes(self):
        _a=pd.read_csv("df_duplicateout.csv")
        #check classification number
        cnt=Counter(_a.Classification)
        print(cnt)
        #calculate the top classes (if there are many classification categories)
        sorted_classes = cnt.most_common()[:5]
        print(sorted_classes)
        #plot the class frequence (will be helpful when there are many classes)
        classes = [c[0] for c in sorted_classes]
        counts = [c[1] for c in sorted_classes] 
        #have a look at the length distribution of the sequences
        seqs = _a.Sequence.values 
        lengths = [len(s) for s in seqs]  
        print(lengths[:20])
        #now create the figures
        fig, axarr = plt.subplots(1,2, figsize=(20,5)) 
        axarr[0].bar(range(len(classes)), counts)
        plt.sca(axarr[0])
        plt.xticks(range(len(classes)), classes, rotation='vertical')
        axarr[0].set_ylabel('frequency')
        axarr[1].hist(lengths, bins=50,density=0)
        axarr[1].set_xlabel('sequence length')
        axarr[1].set_ylabel('#sequences')
        #display the figure
        plt.show()


class MyModel(tf.keras.Model):
    def __init__(self):
        #inherence of the method of keras
        super().__init__()
        #build the model via the Sequantial method
        
    def BinaryY(self, _X): #_X here is the training file "df_duplicateout.csv"
    # Transform labels to one-hot
        self._a = pd.read_csv(_X)  #read the file into a DataFrame
        lb = LabelBinarizer()
        self.Y = lb.fit_transform(self._a.Classification)
        #have a look the binary labels
        print(self.Y,self.Y.shape)
        return self.Y
  
        
    def model_builder(self):
        self._b = self._a.Sequence
        
        #a tokenizer to code the protein sequences as texts
        tokenizer = Tokenizer(char_level=True)
        tokenizer.fit_on_texts(self._b)
        X = tokenizer.texts_to_sequences(self._b)  
        self.X = sequence.pad_sequences(X, maxlen=MaxLen)
        print(self.X)
        
        #build the model using the method Sequenctial()
        embedding_dim = 21
        self.model = Sequential()
        self.model.add(Embedding(len(tokenizer.word_index)+1, embedding_dim, input_length=MaxLen)) 
        self.model.add(Conv1D(filters=128, kernel_size=4, padding='same', activation='relu',dilation_rate=1))
        self.model.add(Conv1D(64, 1, activation='relu'))
        self.model.add(build_model((200,64),head_size=256, num_heads=2, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.2, mlp_dropout=0.2))
#         self.model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(128, activation='relu')) 
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))  
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        return self.model
    
    def call(self):
        # preparation of training datasets
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=.2)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=123)#data Fixed data set structure
        

        Reduce = ReduceLROnPlateau(factor=0.1, min_lr=0.01, patience=10, monitor='val_loss', verbose=1)
        # here the assignment can not be the same as the function name of "ReduceLROnPlateau", otherwise will lead to a local variable error
        self.history = self.model.fit(X_train, y_train, batch_size=128, epochs=25, validation_split=0.15, callbacks=[Reduce])
        self.test_pred = self.model.predict(X_test)
        self.train_pred = self.model.predict(X_train)
        # Save training and testing data
        np.savez_compressed('resultstsnum2/training_data.npz', X_train=X_train, y_train=y_train)
        np.savez_compressed('resultstsnum2/testing_data.npz', X_test=X_test, y_test=y_test)
        
        y_pred_classes = (self.test_pred > 0.5).astype("int32")
        y_pred_classes_train = (self.train_pred > 0.5).astype("int32")
        self.pred_classes_test=y_pred_classes
        self.pred_classes_train=y_pred_classes_train
        self.y_test=y_test
        self.y_train=y_train        
        # Save predicted results
        np.savetxt('resultstsnum2/predicted_results_test.csv', y_pred_classes, delimiter=',')#TEST predicted
        np.savetxt('resultstsnum2/predicted_results_train.csv', y_pred_classes_train, delimiter=',')#TRAIN predicted

        # Save test results
        test_results = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred_classes.flatten()})
        test_results.to_csv('resultstsnum2/test_results.csv', index=False)
        #save train results
        train_results = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_pred_classes_train.flatten()})
        train_results.to_csv('resultstsnum2/train_results.csv', index=False)        
        # 计算召回率和查准率\F1
        recalltest = recall_score(y_test, y_pred_classes)
        precisiontest = precision_score(y_test, y_pred_classes)
        f1test = f1_score(y_test, y_pred_classes)
        recalltrain = recall_score(y_train, y_pred_classes_train)
        precisiontrain = precision_score(y_train, y_pred_classes_train)
        f1train = f1_score(y_train, y_pred_classes_train)

        print("Test Recall:", recalltest)
        print("Test Precision:", precisiontest)
        print("Test F1:", f1test)

        print("Train Recall:", recalltrain)
        print("Train Precision:", precisiontrain)
        print("Train F1:", f1train)
        
        
        # 计算test混淆矩阵
        conf_mat = confusion_matrix(y_test, y_pred_classes)

        # 绘制混淆矩阵热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['MT', 'NoneMT'], yticklabels=['MT', 'NoneMT'])###换其他数据集这里要改进
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        figure_save_path="resultstsnum2"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path , 'confusion-Matrix_test.png'))#第一个是指存储路径，第二个是图片名字
        
        plt.show()

        # 计算train混淆矩阵
        conf_mat_train = confusion_matrix(y_train, y_pred_classes_train)

        # 绘制混淆矩阵热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat_train, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['MT', 'NoneMT'], yticklabels=['MT', 'NoneMT'])###换其他数据集这里要改进
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        figure_save_path="resultstsnum2"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path , 'confusion-Matrix_train.png'))#第一个是指存储路径，第二个是图片名字
        
        plt.show()

        print("test-acc = " + str(accuracy_score(np.argmax(y_test, axis=1), np.argmax(self.test_pred, axis=1))))
        print("train-acc = " + str(accuracy_score(np.argmax(y_train, axis=1), np.argmax(self.train_pred, axis=1))))
        # numpy.argmax(a, axis=None, out=None, *, keepdims=<no value>) Returns the indices of the maximum values along an axis.

        # save the model in the current work directory for next steps
        self.model.save('modelts2')
        
    def plot_prediction(self):
        #visualize the performance of the model
        
        s, (at, al) = plt.subplots(2,1)
        at.plot(self.history.history['accuracy'], c= 'orangered')
        at.plot(self.history.history['val_accuracy'], c='darkviolet')
        #here for history.history the first is self.history object created in call()
        
        at.set_title('Model accuracy')
        at.set_ylabel('Accuracy')
        at.set_xlabel('Epoch')
        at.legend(['MyModel_train', 'MyModel_val'], loc='upper left')
#         figure_save_path="resultsDMP"
#         if not os.path.exists(figure_save_path):
#             os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
#         sns_plot.figure.savefig(os.path.join(figure_save_path , 'accuracy.png'))#第一个是指存储路径，第二个是图片名字
 
        
        al.plot(self.history.history['loss'], c='olive')
        al.plot(self.history.history['val_loss'], c='teal')
        al.set_title('model loss')
        al.set_ylabel('Loss')
        al.set_xlabel('Epoch')
        al.legend(['Train', 'Val'], loc = 'upper left')
        figure_save_path="resultstsnum2"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        s.savefig(os.path.join(figure_save_path , 'acc+loss.png'))#第一个是指存储路径，第二个是图片名字  
        plt.show()
        
    def plt_ROC_curve(self):
        lw=2
        fpr, tpr, thresholds = roc_curve(self.y_test,self.pred_classes_test)
        roc_auc = auc(fpr, tpr)
        plt.subplots(1,1)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Example')
        plt.legend(loc="lower right")
        figure_save_path="resultstsnum2"
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path , 'ROC.png'))#第一个是指存储路径，第二个是图片名字
        plt.show()    
        
    def prot_pred(self, training_data, prediction_data):
        
        seqs = pd.read_csv(training_data).Sequence.values
        pred = pd.read_csv(prediction_data).Sequence.values
        #the csv file was derived from the FASTA protein data
        
        #code the sequences of prediction dataset by fitting it with the tokenizer created for the training data
        tokenizer = Tokenizer(char_level=True) 
        tokenizer.fit_on_texts(seqs)
        word_index = tokenizer.word_index
        print(len(word_index), word_index)
        print(tokenizer.word_counts,tokenizer.word_docs)
        
        #coding the sequences using the tokenizer
        print(pred.describe(), pred.shape)
        
        X_pred_seq  = tokenizer.texts_to_sequences(pred) 
        X_pred_seq = sequence.pad_sequences(X_pred_seq, maxlen=MaxLen) 
        print(X_pred_seq,X_pred_seq.shape)
        
        #loading the model
        model = load_model('modelts2')
        result_predict=model.predict(X_pred_seq,batch_size=5)
        print(result_predict,result_predict.shape)
        
        #save the results
        result_combined = pd.DataFrame(result_predict)
        result_combined.to_csv('result_combined.csv',index=False)  
        #a file naming 'result_combined.csv' can be seen in the current work directory


class ModelPred():
    
    #this class define functions to apply the model for prediction of target proteins from metagenomes using the trained model
    #the protein datasets were annotated coding features from environmental metagenomes, which can be done through another pipeline
    #the result is a matrix indicating the indicating the probability of each protein sequences for belonging to the target protein
    
    def __init__(self,training_data, prediction_data):
        
        self.seqs = pd.read_csv(training_data).Sequence.values
        self.pred = pd.read_csv(prediction_data).Sequence.values
        self._c = pd.read_csv(prediction_data)
        #here the training_data and prediction_data are all csv files from the preprocessing steps. training_data is the duplicateout file, and prediction_data is the predictionset. Both files can be inputs as its absolute address.
    def prot_pred(self):
        #the csv file was derived from the FASTA protein data
        #code the sequences of prediction dataset by fitting it with the tokenizer created for the training data
        tokenizer = Tokenizer(char_level=True) 
        tokenizer.fit_on_texts(self.seqs)
        word_index = tokenizer.word_index
        print(len(word_index), word_index)  
        #have a look at the word index, and this can be checked again later to ensure both dataset applied to the same
        print(tokenizer.word_counts,tokenizer.word_docs)
        
        #coding the sequences using the tokenizer
        #first have a look at the basic information of the prediction dataset
        
        lengths = [len(s) for s in self.pred]
        print(lengths[:20])
        
        X_pred_seq  = tokenizer.texts_to_sequences(self.pred) 
        X_pred_seq = sequence.pad_sequences(X_pred_seq, maxlen=MaxLen) 
        print(X_pred_seq,X_pred_seq.shape)
        
        #loading the model
        model = load_model('model1')
        result_predict=model.predict(X_pred_seq,batch_size=5)
        print(result_predict,result_predict.shape)
        
        #save the results
        self.result_pred = pd.DataFrame(result_predict)
        self.result_pred.to_csv('result_pred.csv',index=False)  
        #a file naming 'result_combined.csv' can be seen in the current work directory
        #now the result is a one-dimention dataframe of probability value lacking of the sequence information
        #if it is a multi-classification, you need to use 'np.argmax(self.result_combined, axis=1)'
        
        print(self.result_pred.head())
        print(self.result_pred.describe())
       
        
    def result_processing(self):
        #now we need to combine the probability value with the sequence info
        #covert the numpy array to DF
        print(self._c)
        #join the two DFs returning a new DF
        _d = self.result_pred.join(self._c.Sequence,how='inner')
        print(_d, type(_d))
        
        _d.to_csv('result_combined.csv')
        
        #convert _d to a numpy array
        result_combined=pd.read_csv('result_combined.csv')
        _array=np.array(result_combined)
        print(type(_array), _array)
        
        
        #filter the results
        _e=[]
        for item in _array:
            datapoint=item[1]
            if datapoint<CutOffValue:  #we choose a threshole value of 0.001 to filter out the non-targe sequences
                print(item)
                _e.append(item)
        print(type(_e))
        name=['index','probability','sequence']
        result_final=pd.DataFrame(columns=name, data=_e)
        result_final.to_csv('result_combined_CutOffValue.csv')
        #if the predicted sequences are too many for subsequence processing , reduce the threshold value to 0.000001
        
        
       
        
        

        








