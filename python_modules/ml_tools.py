#!/usr/bin/env python
# coding: utf-8
# Author: Xianglin Wu (xianglin3092@gmail.com)

# algorithm tools
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF

from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

# seperate data tool
from sklearn.model_selection import train_test_split

# evaluate model tools
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from utilis import read_yaml

class ML_SET:
    def __init__(self, Xdata, Ydata, predictX=None, predict_data=None):
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.predictX = predictX
        self.predict_data = predict_data
        self.algorithm_base = read_yaml('../support_algorithm.yaml')
        self.algorithm_types = list( self.algorithm_base.keys() )
        self.used_algorithm = []
    
    def activate(self, algorithm_list):
        '''
        Create ML_ACT objects and execute the algorithms which user selected.
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        if ( len(algorithm_list) == 1 ) and ( algorithm_list[0]=='all' ):
            for algorithm in self.algorithm_base:
                self.__dict__[algorithm] = ML_ACT(self.Xdata, self.Ydata, self.predictX, self.predict_data, algorithm)
                self.__dict__[algorithm].start_ml()
                self.used_algorithm.append(algorithm)
            msg = 'Activate successfully'
            return msg

        elif set(algorithm_list).issubset(self.algorithm_base):
            for algorithm in algorithm_list:
                self.__dict__[algorithm] = ML_ACT(self.Xdata, self.Ydata, self.predictX, self.predict_data, algorithm)
                self.__dict__[algorithm].start_ml()
                self.used_algorithm.append(algorithm)
                print(algorithm)
            msg = 'Activate successfully'
            return msg

        else:
            msg = 'Some elements you insert are not supportable or wrong.'
            return msg

    def export_ml_obj(self, algorithm_list):
        '''
        Export pkl files for certain algorithms
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        if ( len(algorithm_list) == 1 ) and ( algorithm_list[0]=='all' ):
            for algorithm in self.used_algorithm:
                pickle.dump(self.__dict__[algorithm].ml_obj, open('../output_data/pkl/%s.pkl'%(algorithm), 'wb'))
            msg = 'Export ML OBJ successfully'
            return msg

        elif set(algorithm_list).issubset(self.used_algorithm):
            for algorithm in algorithm_list:
                pickle.dump(self.__dict__[algorithm].ml_obj, open('../output_data/pkl/%s.pkl'%(algorithm), 'wb'))
            msg = 'Export ML OBJ successfully'
            return msg
                
        else:
            msg = 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.'
            return msg
    
    def merge_models(self, algorithm_list):
        '''
        Marge certain training and testing results for selected algorithms in a DataFrame.
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        if ( len(algorithm_list) == 1 ) and ( algorithm_list[0]=='all' ):
            merge_list = []
            for algorithm in self.used_algorithm:
                merge_list.append(self.__dict__[algorithm].models)
            self.merged_models_df = pd.concat(merge_list, axis=0)
            msg = 'Merge models successfully'
            return msg

        elif set(algorithm_list).issubset(self.used_algorithm):
            merge_list = []
            for algorithm in algorithm_list:
                merge_list.append(self.__dict__[algorithm].models)
                self.merged_models_df = pd.concat(merge_list, axis=0)
            msg = 'Merge models successfully'
            return msg
            
        else:
            msg = 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.'
            return msg
    
    def remove_models(self, algorithm_list):
        '''
        Remove certain training and testing results for selected algorithms from a DataFrame.
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        if ( len(algorithm_list) == 1 ) and ( algorithm_list[0]=='all' ):
            self.merged_models_df = None
            msg = 'Remove models successfully'
            return msg

        elif set(algorithm_list).issubset(self.used_algorithm):
            for algorithm in algorithm_list:
                del self.merged_models_df.loc['Train'==self.algorithm_base[algorithm]]
            msg = 'Remove models successfully'
            return msg
            
        else:
            msg = 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.'
            return msg

    def export_models(self):
        '''
        Export the DataFrame of training and testing results for selected algorithms as a csv file.
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        self.merged_models_df.to_csv('../output_data/綜合表.csv',index=False)
        msg = 'Export models successfully'
        return msg
    
    def export_predict(self, algorithm_list):
        '''
        Export the DataFrame of predicting results for selected algorithms as a csv file.
        Input: 
            algorithm_list: list

        Output:
            msg: str
        '''
        if ( len(algorithm_list) == 1 ) and ( algorithm_list[0]=='all' ):
            for algorithm in self.used_algorithm:
                predict_dataframe = pd.concat([self.predict_data, self.__dict__[algorithm].predict_result],axis=1)
                predict_dataframe.sort_values('時間', inplace=True, ascending=True)
                predict_dataframe.to_csv('../output_data/predict_result/%s_predict_result.csv'%(algorithm),index=False)
            msg = 'Export prediction results successfully'
            return msg

        elif set(algorithm_list).issubset(self.used_algorithm):
            for algorithm in algorithm_list:
                predict_dataframe = pd.concat([self.predict_data, self.__dict__[algorithm].predict_result],axis=1)
                predict_dataframe.sort_values('時間', inplace=True, ascending=True)
                predict_dataframe.to_csv('../output_data/predict_result/%s_predict_result.csv'%(algorithm),index=False)
            msg = 'Export prediction results successfully'
            return msg
            
        else:
            msg = 'Some elements you insert are not supportable or wrong.\nBesides, chances are some algorithms may not be activated.'
            return msg





class ML_ACT(ML_SET):
    def __init__(self, Xdata, Ydata, predictX=None, predict_data=None, algorithm=None):
        super().__init__(Xdata, Ydata, predictX, predict_data)
        self.algorithm = algorithm
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.Xdata, self.Ydata.values , test_size=0.3, random_state=20)

    def algorithm_select(self):
        '''
        This is the area where users can adjust the parameters of algorithm objects.
        '''
        if self.algorithm == 'knn':
            obj = KNeighborsClassifier(n_neighbors = 3)

        elif self.algorithm == 'svms':
            obj = SVC()

        elif self.algorithm == 'lsvc':
            obj = LinearSVC()
        
        elif self.algorithm == 'br':
            obj = BayesianRidge()
        
        elif self.algorithm == 'gls':
            obj = LinearRegression()
        
        elif self.algorithm == 'logir':
            obj = LogisticRegression(random_state=0)
        
        elif self.algorithm == 'gpr':
            kernel = DotProduct() + WhiteKernel()
            obj = GaussianProcessRegressor(kernel=kernel,random_state=0)
            # kernel = 1.0 * RBF(1.0)
            # obj = GaussianProcessClassifier(kernel=kernel,random_state=0)
        
        elif self.algorithm == 'pla':
            obj = Perceptron(tol=1e-3, random_state=0)
        
        elif self.algorithm == 'sgd':
            obj = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
        
        elif self.algorithm in ['dt3', 'dt4', 'dt5']:
            depth = int(self.algorithm[-1])
            obj = DecisionTreeClassifier(max_depth=depth)
        
        elif self.algorithm == 'gauNB':
            obj = GaussianNB()
        
        elif self.algorithm == 'berNB':
            obj = BernoulliNB()
        
        elif self.algorithm == 'comNB':
            obj = ComplementNB()
        
        elif self.algorithm == 'mulNB':
            obj = MultinomialNB()
        
        elif self.algorithm == 'cateNB':
            obj = CategoricalNB()
        
        return obj

    def start_ml(self):
        '''
        Execute the selected algorithm, generate a testing and training result and a confusion matrix.
        '''
        self.ml_obj = self.algorithm_select()
        self.ml_obj.fit(self.X_train, self.Y_train)
        self.Y_predict_t = self.ml_obj.predict(self.X_test)
        if self.algorithm in ['gls','gpr','br']:
            std,mean = self.Y_predict_t.std(),self.Y_predict_t.mean()
            for num in range(len(self.Y_predict_t)):
                if mean+std < self.Y_predict_t[num] < mean+std*2:
                    self.Y_predict_t[num] = 1
                elif self.Y_predict_t[num] > mean+std*2:
                    self.Y_predict_t[num] = 2
                else:
                    self.Y_predict_t[num] = 0
            
            self.Y_predict_t = np.array(["%.0f" % w for w in self.Y_predict_t.reshape(self.Y_predict_t.size)])
        self.generate_models()
        self.generate_confusion_matrix()
        if self.predictX:
            self.generate_predict()
        
    def generate_models(self):
        '''
        Generate a testing and training result.
        '''
        acc = round( self.ml_obj.score(self.X_train, self.Y_train) * 100, 2 )
        acc_test = round( self.ml_obj.score(self.X_test, self.Y_test) * 100, 2 )
        precision = round( precision_score(self.Y_test, self.Y_predict_t, average="macro") * 100, 2 )
        recall = round( recall_score(self.Y_test, self.Y_predict_t, average="macro") * 100, 2 )
        fscore = round( f1_score(self.Y_test, self.Y_predict_t, average="macro") * 100, 2 )
        self.models = pd.DataFrame({'Train': [ self.algorithm_base[self.algorithm] ],      
                                    'accuracy': [acc],
                                    'test': [acc_test],
                                    'precision': [precision],
                                    'recall': [recall],
                                    'fscore': [fscore] })
        self.models.sort_values(by='accuracy', ascending=False)
    
    def generate_confusion_matrix(self):
        '''
        Generate a confusion matrix.
        '''
        cm  = confusion_matrix(self.Y_test, self.Y_predict_t) 
        cms = sns.heatmap(cm, square=True, annot=True, cbar=False, fmt='g')
        cms.set_title('demo_gnb_confusion_matrix(%s)'%(self.algorithm))
        cms.set_xlabel("predicted value")
        cms.set_ylabel("true value")
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.savefig('../output_data/images/%s_confusion_matrix.png'%(self.algorithm), dpi=200)
        plt.show()
        plt.close()
    
    def generate_predict(self):
        '''
        Generate a predicting result.
        '''
        self.Y_predict_p = self.ml_obj.predict(self.predictX)
        if self.algorithm in ['gls','gpr','br']:
            ##change prob to label##
            std,mean = self.Y_predict_p.std(),self.Y_predict_p.mean()
            for num in range(len(self.Y_predict_p)):
                if mean + std < self.Y_predict_p[num] < mean + std * 2:
                    self.Y_predict_p[num] = 1
                elif self.Y_predict_p[num] > mean + std * 2:
                    self.Y_predict_p[num] = 2
                else:
                    self.Y_predict_p[num] = 0
            self.Y_predict_p = np.array(["%.0f" % w for w in self.Y_predict_p.reshape(self.Y_predict_p.size)])
        
        self.predict_result = pd.DataFrame(self.Y_predict_p)
        self.predict_result.rename( columns={0: '%s_predict_result'%(self.algorithm)}, inplace=True)