from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import bisect
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from fuzzytrees.frdf import BaseFuzzyRDF
from fuzzytrees.fdts import FuzzyCARTClassifier,FuzzyID3Classifier, FuzzyC45Classifier
from fuzzytrees.util_tree_criterion_funcs import majority_vote
from fuzzytrees.fdt_base import FuzzyDecisionTreeWrapper
import json

"""
Cautious Classifiers

The below classes implement a framework for cautious classification. The goal is to design classifiers that 
can abstain and or are less likley to make decisions when there is high uncertainty. It includes multiple cautious 
classification strategies. 

Key Classes:
1. CautiousClassifier (Abstract Base Class):
   - Defines the structure for cautious classifiers.
   - Enforces implementation of `fit` and `predict` methods in derived classes.

2. NaiveCautiousClassifier:
   - A baseline implementation using a Random Forest Classifier.
   - Implements threshold-based abstention, where predictions are discarded if the confidence is below a user-defined threshold.
   - Includes functionality for hyperparameter tuning using GridSearchCV.

3. ConformalPredictor:
   - A probabilistic classifier leveraging conformal prediction.
   - Generates prediction sets with a predefined coverage guarantee, abstaining when confidence intervals overlap.
   - Supports calibration with nonconformity scores for robust decision-making.

4. WCRF (Weighted Conformal Random Forest):
   - Extends Random Forest with region-specific predictions and weights.
   - Combines probabilistic predictions using weighted intervals and abstains when predictions are ambiguous.
   - Includes parameter tuning and optimization for improved performance.

5. RandomForest:
   - A simple wrapper around scikit-learn's RandomForestClassifier.
   - Implements hyperparameter tuning, fitting, and cross-validation.
"""


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
   
class NaiveCautiousClassifier(CautiousClassifier):
    """
    This class is implemented based on ideas and code from the below source. 
    Reference:
    https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
    """
    def __init__(self, X, y, threshold):
        self.data = X
        self.labels = y
        self.threshold = threshold

    def gridSearch(self):
        clf=RF(random_state=0)
        param_grid = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20, 30],     
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(self.data, self.labels)
        return grid_search.best_params_
    
    def fit(self, best_params):
        self.model = RF(**best_params, random_state=0)
        self.model.fit(self.data, self.labels)


    def predict(self, X_test):
        preds = self.model.predict_proba(X_test)  
        max_probs = np.max(preds, axis=1)               
        predicted_labels = np.argmax(preds, axis=1)
        predicted_labels = np.where(max_probs >= self.threshold, predicted_labels, -1)

        return predicted_labels
    
    def predict_x_val(self, X_test,y_test,cv=10):
        cvscore=cross_val_score(self.model,X_test,y_test,cv)
        return cvscore 
    
## NOT WORKING SO COMMENTING OUT
# class CostSensetiveCautiousClassifier(CautiousClassifier):
#     """
#     This class is implemented based on ideas and code from the below source. 
#     Reference:
#     https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
#     """
#     def __init__(self, X, y, costs):
#         self.data = X
#         self.labels = y
#         self.costs = costs

#     def gridSearch(self):
#         clf=RF(random_state=0)
#         param_grid = {
#         'n_estimators': [50, 100, 200], 
#         'max_depth': [None, 10, 20, 30],     
#         'max_features': ['sqrt', 'log2', None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#         grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
#         grid_search.fit(self.data, self.labels)
#         return grid_search.best_params_
    
#     def fit(self, best_params):
#         self.model = RF(**best_params, random_state=0)
#         self.model.fit(self.data, self.labels)

#     def get_cost(self, probs, y_calib, thresholds):
#         for i in range(len(probs)):
#             prob = probs[i]
#             above_thresh = prob >= thresholds

#             if not np.any(above_thresh):
#                 total_cost += self.cost['abstention']
#             else:
#                 max_prob = np.max(prob, axis=1)
#                 class_cost = 

#     def fit_cost_sensetive_threshold(self, X_calib, y_calib, search_grid=np.linspace(0.5, 0.99, 10)):

#         probs = self.model.predict_proba(X_calib)
#         thresholds = np.full(2, 0.5)
#         for label in range(2):
#             best = thresholds[label]
#             min_cost = self.get_cost(probs, y_calib, thresholds)
#             for thresh in search_grid:
#                 test= thresholds.copy()
#                 test[label] = thresh
#                 cost = self.get_cost(probs, y_calib, thresholds)
#                 if cost < min_cost:
#                     min_cost = cost
#                     best = thresh
#             thresholds[label] = best
#         self.thresholds = thresholds

#     def internal_predict(self, X_test):
#         preds = self.model.predict_proba(X_test)  
#         preds_return = np.full(len(preds), -1)
#         for i in range(len(preds)):
#             above_thresh = preds[i] >= self.thresholds
#             if not np.any(above_thresh):
#                 continue
#             else:
#                 max_prob = np.max(preds[i], axis=1)               
#                 predicted_label = above_thresh[np.argmax(preds[i], axis=1)]
#                 preds_return[i] =predicted_label

#         return preds_return
    

    
#     def predict_x_val(self, X_test,y_test,cv=10):
#         cvscore=cross_val_score(self.model,X_test,y_test,cv)
#         return cvscore 

    
class ClassSpecificThresholdsCautiousClassifier(CautiousClassifier):
    """
    This class is implemented based on ideas and code from the below source. 
    Reference:
    https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
    """
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def gridSearch(self):
        clf = RF(random_state=0)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(self.data, self.labels)
        return grid_search.best_params_

    def fit(self, best_params):
        self.model = RF(**best_params, random_state=0)
        self.model.fit(self.data, self.labels)

    def internal_eval(self, thresh, preds_proba, y_calib):
            y_pred = self._internal_predict(preds_proba, thresh)
            # count = (y_pred != -1)
            imprecise_predictions = y_pred
            indeterminate_instance = (imprecise_predictions == -1)
            determinate_instance = (imprecise_predictions != -1)
            single_set_length = len(y_calib) - sum(indeterminate_instance)
            determinacy = single_set_length/len(y_calib)
            determinacy = round(determinacy*100, 2)
            single_set_accuracy = sum(y_calib[determinate_instance]==imprecise_predictions[determinate_instance])/single_set_length
            single_set_accuracy = round(single_set_accuracy*100, 2)
            u65_score = round(65 + (single_set_accuracy - 65)*determinacy/100, 2)
            return u65_score


    def fit_class_thresholds(self, X_calib, y_calib, search_grid=np.linspace(0.0, 1.0, 11)):
        preds_proba = self.model.predict_proba(X_calib)
        thresholds = np.full(2, 0.5)

        for label in range(2):
            best_thresh = thresholds[label]
            best_score = self.internal_eval(thresholds, preds_proba, y_calib)
            for candidate_threshold in search_grid:
                test_thresh = thresholds.copy()
                test_thresh[label] = candidate_threshold
                score = self.internal_eval(test_thresh, preds_proba, y_calib)
                if score > best_score:
                    best_score = score
                    best_thresh = candidate_threshold
            thresholds[label] = best_thresh

        self.class_thresholds = thresholds

    def predict(self, X_test):
        preds = self.model.predict_proba(X_test)
        return self._internal_predict(preds, self.class_thresholds)

    def _internal_predict(self, preds, thresh):
        predictions = []
        for p in preds:
            qualified = p >= thresh
            if not np.any(qualified):
                predictions.append(-1)
            else:
                ratios = p[qualified] / thresh[qualified]
                qualified_classes = np.where(qualified)[0]
                chosen_class = qualified_classes[np.argmax(ratios)]
                predictions.append(chosen_class)
        return np.array(predictions)

    def predict_x_val(self, X_test, y_test, cv=10):
        cvscore = cross_val_score(self.model, X_test, y_test, cv=cv)
        return cvscore


class ConformalPredictor(CautiousClassifier, ):
    """
    This class is implemented based on ideas and code from the below source. 
    Reference:
    https://blog.dataiku.com/measuring-models-uncertainty-conformal-prediction
    """

    def __init__(self,n_trees=100, s=2, gamma=1, labda=1, tree_max_depth=None, combination=1, data_name=None, random_state=None):
        self.n_trees = n_trees
        self.s = s
        self.labda = labda
        self.gamma = gamma
        self.combination = combination
        self.data_name = data_name
        self.w = np.ones(n_trees) / n_trees
        # self.model = RF(best_params, random_state=random_state)
    def gridSearch(self, X, y):
        clf=RF(random_state=0)
        param_grid = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20, 30],     
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def fit(self,best_params, X_train, y_train, X_calib, y_calib):
        self.model = RF(**best_params, random_state=0)
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
    """
    This class is from the below source. 
    Reference:
    https://github.com/Haifei-ZHANG/cf_for_wcrf/blob/main/wcrf.py
    """
    def __init__(self, n_trees=100, s=2, gamma=1, labda=1, tree_max_depth=None, combination=1, data_name=None, random_state=None):
        # build a random forest using sklearn RandomForestClassifier
        self.n_trees = n_trees
        self.s = s
        self.labda = labda
        self.gamma = gamma
        self.combination = combination
        self.data_name = data_name
        self.w = np.ones(n_trees)/n_trees
        self.model = RF(n_estimators=n_trees, max_depth=tree_max_depth, random_state=random_state)
        
        
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
    
    def gridSearch(self, X_train, y_train, X_test, y_test):
        opt=0;opt_s=0;opt_gam=0;opt_lam=0
        for i in range(1,10):
            for j in range(1,10):
                for k in range(1,10):
                    self.s = i
                    self.gamma = j
                    self.labda = k
                    self.fit(X_train,y_train)
                    self.fit_w(X_train,y_train)
                    preds=self.predict(X_test)[0]
                    if np.mean(preds==y_test)>opt:
                        opt_s=i; opt_gam=j; opt_lam=k
                        opt=np.mean(preds==y_test)
        self.s = opt_s
        self.gamma = opt_gam
        self.labda = opt_lam

class RandomForest:
    def __init__(self, X, y):
        self.data = X
        self.labels = y

    def gridSearch(self):
        clf=RF(random_state=0)
        param_grid = {
        'n_estimators': [50, 100, 200], 
        'max_depth': [None, 10, 20, 30],     
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
        grid_search.fit(self.data, self.labels)
        return grid_search.best_params_
    
    def fit(self, best_params):
        self.model = RF(**best_params, random_state=0)
        self.model.fit(self.data, self.labels)


    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_x_val(self, X_test,y_test,cv=10):
        cvscore=cross_val_score(self.model,X_test,y_test,cv)
        return cvscore 


class FuzzyRandomForest(BaseFuzzyRDF):
    def __init__(self, disable_fuzzy, fuzzification_options, criterion_func, n_estimators=100,
                 max_depth=3, min_samples_split=2, min_impurity_split=1e-7, max_features=None,
                 multi_process_options=None,fdt_class=FuzzyCARTClassifier):
        self.fdt_class = fdt_class
        self.best_options = {}
        super().__init__(disable_fuzzy=disable_fuzzy,
                         fuzzification_options=fuzzification_options,
                         criterion_func=criterion_func,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_impurity_split=min_impurity_split,
                         max_features=max_features,
                         multi_process_options=multi_process_options)

        # Initialise the forest.
        self.build_estimators()

        # Specify to get the final classification result by majority voting method.
        self._res_func = majority_vote
    
    def build_estimators(self):
        self._estimators = []
        for _ in range(self.n_estimators):
            estimator = FuzzyDecisionTreeWrapper(fdt_class=self.fdt_class,
                                                 disable_fuzzy=self.disable_fuzzy,
                                                 fuzzification_options=self.fuzzification_options,
                                                 criterion_func=self.criterion_func,
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 min_impurity_split=self.min_impurity_split)
            self._estimators.append(estimator)


    def fit(self,X_train, y_train):
        X_train = X_train.to_numpy()
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        
        super().fit(X_train,y_train)

    def predict(self,X_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        return np.array(super().predict(X_test))

    def gridSearch(self,dataset_name,X_train, y_train, X_test, y_test):
        fname = 'frf_param_config.json'
        opt = 0
        n_estimators = [50,100,200]
        depths = [1,10,20,30]
        features = ['sqrt', 'log2', None]
        min_samples_split = [2, 5, 10]
        impurity_splits = [1e-7,1e-6,1e-5]
        classes = [FuzzyCARTClassifier]
        fuzz_options = [5]
        opt_cl = None
        opt_est = 0
        opt_depth = 0
        opt_mss = 0
        opt_mis = 0
        
        for est in n_estimators:
            self.n_estimators = est
            for cl in classes:
                self.fdt_class = cl
                for depth in depths:
                    self.max_depth = depth
                    for split in min_samples_split:
                        self.min_samples_split = split
                        for imp in impurity_splits:
                            self.min_impurity_split = imp
                            self._estimators = []
                            self.build_estimators()
                            self.fit(X_train,y_train)
                            preds=self.predict(X_test)
                            if np.mean(preds==y_test)>opt:
                                    opt_cl = cl
                                    opt_est = est
                                    opt_depth = depth
                                    opt_mss = split
                                    opt_mis = imp
                                    opt = np.mean(preds == y_test)

        self.fdt_class = opt_cl
        self.n_estimators = opt_est
        self.max_depth=opt_depth
        self.min_samples_split = opt_mss
        self.min_impurity_split = opt_mis
        self.build_estimators()

        opts = {}
        opts['n_estimators'] = opt_est
        opts['max_depth'] = opt_depth
        opts['min_samples_split'] = opt_mss
        opts['min_impurity_split'] = opt_mis
        if opt_cl == FuzzyCARTClassifier:
            opts['fdt_class'] = 'CART'
        elif opt_cl == FuzzyID3Classifier:
            opts['fdt_class'] = 'ID3'
        else:
            opts['fdt_class'] = 'C45'
        self.best_options[dataset_name] = opts
        curr = {}
        with open(fname,'r+') as f:
            curr = json.load(f)
        curr[dataset_name] = opts
        with open(fname, 'w+') as f:
            json.dump(curr,f)
       

   