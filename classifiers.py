from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import bisect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.special import expit
from scipy.optimize import minimize


class CautiousClassifier(ABC):

    def __init__(
        self,
        model=None,
        target_label=None,
        name=None
    ):
        super().__init__()
        self.model = model
        self.name = name            
    
    def fit(self, X_train, y_train):
        raise NotImplementedError()

    def predict(self, X_test):
        ''''''
        raise NotImplementedError()
   
class ConformalPredictor(CautiousClassifier):
    def __init__(self, n_trees=100, s=2, gamma=1, labda=1, tree_max_depth=None, combination=1, data_name=None, random_state=None):
        self.n_trees = n_trees
        self.s = s
        self.labda = labda
        self.gamma = gamma
        self.combination = combination
        self.data_name = data_name
        self.w = np.ones(n_trees) / n_trees
        self.model = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_max_depth, random_state=random_state)

    def fit(self, X_train, y_train, X_calib, y_calib):
        self.model.fit(X_train, y_train)

        # Store class information
        self.classes = self.model.classes_
        self.class_to_index = {c: i for i, c in enumerate(self.classes)}

        # calibration nonconformity scores
        # for each sample:
        # nonconformity_score = 1 - p(correct_class|x_calib)
        prob_calib = self.model.predict_proba(X_calib)
        self.calibration_scores_by_class = {c: [] for c in self.classes}

        for i, true_class in enumerate(y_calib):
            class_idx = self.class_to_index[true_class]
            # nonconformity score for this instance is 1 - probability of the true class
            nonconformity_score = 1 - prob_calib[i, class_idx]
            self.calibration_scores_by_class[true_class].append(nonconformity_score)

        for c in self.classes:
            self.calibration_scores_by_class[c] = np.sort(self.calibration_scores_by_class[c])

        return self

    def _nonconformity_score_for_class(self, X, class_idx):
        prob = self.model.predict_proba(X)
        return 1 - prob[:, class_idx]

    def predict_proba(self, X):
        # compute the conformal p-values for each class
        # for a given test point, p-value for class c_j is:

        n_test = X.shape[0]
        p_values = np.zeros((n_test, len(self.classes)))

        for j, c in enumerate(self.classes):
            test_scores = self._nonconformity_score_for_class(X, j)
            sorted_scores = self.calibration_scores_by_class[c]
            N_j = len(sorted_scores)

            # for each test sample, count num calibration scores are >= test_score
            # p-value = (count + 1)/(N_j + 1)
            for i, score in enumerate(test_scores):
                idx = bisect.bisect_left(sorted_scores, score)
                count = N_j - idx
                p_values[i, j] = (count + 1) / (N_j + 1) if N_j > 0 else 1.0

        return p_values

    def predict(self, X, alpha=0.05):
        # get conformal set of classes whose p-value > alpha
        p_values = self.predict_proba(X)
        # thresh at alpha to get sets of classes
        conformal_sets = (p_values > alpha).astype(int)
        mlb = MultiLabelBinarizer(classes=self.classes)
        mlb.fit([self.classes])
        pred = mlb.inverse_transform(conformal_sets)

        return pred
    
class WCRF:
    def __init__(self, n_trees=100, s=2, gamma=1, labda=1, tree_max_depth=None, combination=1, data_name=None, random_state=None):
        # build a random forest using sklearn RandomForestClassifier
        self.n_trees = n_trees
        self.s = s
        self.labda = labda
        self.gamma = gamma
        self.combination = combination
        self.data_name = data_name
        self.w = np.ones(n_trees)/n_trees
        self.model = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_max_depth, random_state=random_state)
        
        
    def fit(self, X, y):
        # fit the model with training set
        self.model.fit(X,y)
        self.classes = self.model.classes_
        self.n_classes = len(self.classes)

        
        # calculate number of sample in each leave for every tree
        trees = self.model.estimators_
        self.leaves_sample_count = []
        for tree in trees:
            leaves_dict = {}
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            value = tree.tree_.value.reshape((-1, self.n_classes))
            for i in range(n_nodes):
                is_leaf = (children_left[i] == children_right[i])
                if is_leaf:
                    leaves_dict[i] = value[i]
                    
            self.leaves_sample_count.append(leaves_dict)
        
        # get regions
        regions = self.model.apply(X)
#         regions = list(set(tuple(region) for region in regions))
        
        # create sample number counter dictionary for each region
        self.regions_sample_count = dict()
        sample_count_array = np.zeros((len(regions), self.n_trees,self.n_classes))
        
        # initialize sample counter for each region, the key is a region presented by tuple, the value is a ndarray
        for i in range(len(regions)):
            region = tuple(regions[i])
            self.regions_sample_count[region] = np.zeros((self.n_trees, self.n_classes))   
            
        # build sample counter for each region
            for t in range(self.n_trees):
                self.regions_sample_count[region][t] = self.leaves_sample_count[t][region[t]]
                sample_count_array[i][t] = self.leaves_sample_count[t][region[t]]

        self.regions_pred_info = dict()

        return 
        

    def fit_w(self, X, y):
        alpha = 10
        beta = 2
        p_intervals = []
        # get falling leaves for each sample in every tree, return array (n_sample * number_trees)
        valid_regions = self.model.apply(X)
        
        for i in range(len(valid_regions)):
            region=tuple(valid_regions[i])
            if region not in self.regions_pred_info.keys():
                self.regions_sample_count[region] = np.zeros((self.n_trees, self.n_classes))
                for t in range(self.n_trees):
                    self.regions_sample_count[region][t] = self.leaves_sample_count[t][region[t]]
                self.regions_pred_info[region] = self.treat_region(region)
            pred_info = self.regions_pred_info[region]
            p_intervals.append(pred_info[3])
            
        p_intervals = np.array(p_intervals)
        p_infs = p_intervals[:,:,0]
        p_sups = p_intervals[:,:,1]

        if self.combination==1:
            k_under = (p_infs>=0.5)+0
            k_over = (p_sups>0.5)+0
        if self.combination==2:
            k_under = p_infs
            k_over = p_sups
        
        def cost_func(w, labda=self.labda, gamma=self.gamma):
            bels = (k_under*w).sum(axis=1)
            pls = (k_over*w).sum(axis=1)
            u_under = expit(alpha*(bels- 0.5))
            u_over = expit(alpha*(pls - 0.5))
            u = expit(beta*(bels - 0.5)*(pls - 0.5))

            cost = -sum(y*np.log(u_under+0.0001) + (1-y)*np.log(1-u_over+0.0001) + gamma*np.log(1-u+0.0001))/len(y) + 0.5*labda*sum(w**2)

            return cost


        def jac(w,labda=self.labda, gamma=self.gamma):
            bels = (k_under*w).sum(axis=1)
            pls = (k_over*w).sum(axis=1)
            u_under = expit(alpha*(bels- 0.5))
            u_over = expit(alpha*(pls - 0.5))
            u = expit(beta*(bels - 0.5)*(pls - 0.5))
            
            der_1 = -alpha*(y*(1-u_under))@k_under
            der_2 = alpha*((1-y)*u_over)@k_over
            der_3 = beta*(k_under.T@(u.reshape((len(y),1))*k_over) + k_over.T@(u.reshape((len(y),1))*k_under))@w - 0.5*u@(k_under+k_over)

            der_j = (der_1 + der_2 + gamma*der_3)/len(y) + labda*w
            
            return der_j
        
        cons = [{'type': 'eq', 'fun':lambda w: sum(w)-1}]

        bounds = [(0,1)] * self.n_trees
        
        w0 = np.random.rand(self.n_trees)
        res = minimize(cost_func, w0, method='SLSQP',jac=jac, constraints=cons, bounds=bounds)
        self.w = res.x
        return
        
        
    def treat_region(self, region):
        sample_count = self.regions_sample_count[region]
        sample_count[:,0] = sample_count[:,0] + sample_count[:,1]
        
        p_intervals = np.zeros_like(sample_count)
        
        p_intervals[:,0] = sample_count[:,1]/(sample_count[:,0] + self.s)
        p_intervals[:,1] = (sample_count[:,1] + self.s)/(sample_count[:,0] + self.s)

        
        # predict
        if self.combination==1:
            k_under = (p_intervals[:,0]>=0.5)+0
            k_over = (p_intervals[:,1]>0.5)+0
            bel = round(sum(k_under*self.w),4)
            pl = round(sum(k_over*self.w),4)
            
        if self.combination==2:
            bel = round(sum(p_intervals[:,0]*self.w),4)
            pl = round(sum(p_intervals[:,1]*self.w),4)
        
        if bel >= 0.5:
            prediction = self.classes[1]
        elif pl <= 0.5:
            prediction = self.classes[0]
        else:
            prediction = -1

        return (prediction, bel, pl, p_intervals, sample_count)

        
    def predict(self, X, plot=False):
        count = 0
        # intitialize prediciton list
        predictions = np.zeros(len(X))
        pred_intervals = []
        p_intervals = []
        # get falling leaves for each sample in every tree, return array (n_sample * number_trees)
        test_regions = self.model.apply(X)
        
        for i in range(len(test_regions)):
            region=tuple(test_regions[i])
            if region not in self.regions_pred_info.keys() or True:
                self.regions_sample_count[region] = np.zeros((self.n_trees, self.n_classes))
                for j in range(self.n_trees):
                    self.regions_sample_count[region][j] = self.leaves_sample_count[j][region[j]]
                self.regions_pred_info[region] = self.treat_region(region)
#                 pred_info = self.treat_region(region)
                count += 1
            pred_info = self.regions_pred_info[region]
            
            #print('bel=',pred_info[1],'pl=',pred_info[2],'pre=',pred_info[0])
            predictions[i] = pred_info[0]
            pred_intervals.append([pred_info[1], pred_info[2]])
            p_intervals.append(pred_info[3])
            
        return predictions, pred_intervals, p_intervals

def vanillaRF(X_train,y_train,X_test,y_test):
    clf=RF(random_state=0)
    GCV=GridSearchCV(clf,param_grid={'max_features':['sqrt', 'log2'],
                                     'n_estimators': [50, 100, 200],
                                    'max_depth': [10, 20, 30],
                                    'min_samples_split': [2, 5, 10],
                                    'min_samples_leaf': [1, 2, 4]},
                     scoring='accuracy',cv=5)
    GCV.fit(X_train,y_train)
    max_feats=GCV.best_params_['max_features']
    n_est=GCV.best_params_['n_estimators']
    max_dep=GCV.best_params_['max_depth']
    mss=GCV.best_params_['min_samples_split']
    
    clf=RF(max_features=max_feats,min_samples_split=mss,max_depth=max_dep,n_estimators=n_est,random_state=0)
    clf.fit(X_train,y_train)
    score=clf.score(X_test,y_test)
    cvscore=cross_val_score(clf,X_test,y_test,cv=10)
    
    preds=clf.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(y_test,preds)
    cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                              display_labels = [0,1])
    cm_display.plot()
    return preds, score, cvscore
