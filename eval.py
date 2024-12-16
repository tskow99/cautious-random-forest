from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score

"""
Project Evaluation Framework

This program provides an interface for evaluating our models, specifically focusing on 
cautious classifiers and their unique metrics. 

**Model Evaluation**:
   - Supports evaluation of multiple cautious classification models, including:
     - `ConformalPredictor`
     - `WCRF` (Weighted Conformal Random Forest)
     - `NaiveCautiousClassifier`
     - `RandomForest`

 **Metrics**:
   - **Determinacy**: Proportion of instances where the model provides a confident (determinate) prediction.
   - **Abstention**: Proportion of instances where the model abstains from making a prediction.
   - **Single-Set Accuracy**: Accuracy on the determinate subset of predictions.
   - **u65 Score**: A custom metric that balances accuracy and determinacy.
   - **Precise Accuracy**: Traditional accuracy metric without abstention for comparison.

"""

def evaluate_model(model_name, X_test, y_test,model):
    if model_name == 'ConformalPredictor':
        return conformal_pred_eval(X_test, y_test,model)
    elif model_name == 'WCRF':
        return wcrf_eval(X_test, y_test,model)
    elif model_name == 'RandomForest':
       return random_forest_eval(X_test, y_test,model) 
    elif model_name == 'NaiveCautiousClassifier':
        return naive_classifier_eval(X_test, y_test,model)
    else:
        raise ValueError(f"Dataset: {model_name} not found")

def naive_classifier_eval(X_test, y_test,model):
    """
    This method is implemented based on ideas and code from the below source. 
    Reference:
    https://github.com/Haifei-ZHANG/cf_for_wcrf/blob/main/wcrf.py
    https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
    """
    y_pred = model.predict(X_test)

    precise_predictions = model.model.predict(X_test)
    precise_accuracy = sum(y_test==precise_predictions)/len(y_test)
    precise_accuracy = round(precise_accuracy*100, 2)

    imprecise_predictions = y_pred
    indeterminate_instance = (imprecise_predictions == -1)
    determinate_instance = (imprecise_predictions != -1)
    # calculate single-set length
    single_set_length = len(y_test) - sum(indeterminate_instance)
        
    # calculate determinacy
    determinacy = single_set_length/len(y_test)
    determinacy = round(determinacy*100, 2)
    
    # calculate single-set accuracy
    single_set_accuracy = sum(y_test[determinate_instance]==imprecise_predictions[determinate_instance])/single_set_length
    single_set_accuracy = round(single_set_accuracy*100, 2)
    
    # claculate u65
    u65_score = round(65 + (single_set_accuracy - 65)*determinacy/100, 2)
    return {'u65_score':u65_score, 
        'single_set_accuracy':single_set_accuracy, 
        'determinacy':determinacy,
        'abstention': 100-determinacy}

def conformal_pred_eval(X_test, y_test,model):
        """
        This method is implemented based on ideas and code from the below source. 
        Reference:
        https://github.com/Haifei-ZHANG/cf_for_wcrf/blob/main/wcrf.py
        https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
        """
        # TO DO IMPLEMENT PRECISE PRED
        # precise_predictions = model.predict(y_test)
        # precise_accuracy = sum(y_test==precise_predictions)/len(y_test)
        # precise_accuracy = round(precise_accuracy*100, 2)
        y_pred = model.predict(X_test)
        y_preds_transformed = [
        -1 if p == (0, 1) else p[0] 
        for p in y_pred
        ]
        y_preds_transformed = np.array(y_preds_transformed)
        precise_predictions = model.model.predict(X_test)
        precise_accuracy = sum(y_test==precise_predictions)/len(y_test)
        precise_accuracy = round(precise_accuracy*100, 2)

        imprecise_predictions = y_preds_transformed
        indeterminate_instance = (imprecise_predictions == -1)
        determinate_instance = (imprecise_predictions != -1)
        
        # calculate single-set length
        single_set_length = len(y_test) - sum(indeterminate_instance)
        
        # calculate determinacy
        determinacy = single_set_length/len(y_test)
        determinacy = round(determinacy*100, 2)
        
        # calculate single-set accuracy
        single_set_accuracy = sum(y_test[determinate_instance]==imprecise_predictions[determinate_instance])/single_set_length
        single_set_accuracy = round(single_set_accuracy*100, 2)
        
        # claculate u65
        u65_score = round(65 + (single_set_accuracy - 65)*determinacy/100, 2)
        return {'u65_score':u65_score, 
            'single_set_accuracy':single_set_accuracy, 
            'determinacy':determinacy,
            'abstention': 100-determinacy}


def wcrf_eval(X_test, y_test,model,  plot=False, show_confusion_matrix=False):
    """
    This method is implemented based on ideas and code from the below source. 
    Reference:
    https://github.com/Haifei-ZHANG/cf_for_wcrf/blob/main/wcrf.py
    https://dmip.webs.upv.es/ROCAI2004/papers/04-ROCAI2004-Ferri-HdezOrallo.pdf
    """
        # get both imprecise and precise predictions 
    imprecise_predictions ,pred_intervals, p_intervals = model.predict(X_test, y_test)
    precise_predictions = np.zeros(len(y_test))
    if model.combination==2:
        precise_predictions = model.model.predict(X_test)
    else:
        for tree in model.model.estimators_:
            precise_predictions += tree.predict(X_test)
        precise_predictions /= model.n_trees
        precise_predictions[precise_predictions>=0.5] = model.classes[1]
        precise_predictions[precise_predictions<0.5] = model.classes[0]
    
    indeterminate_instance = (imprecise_predictions == -1)
    determinate_instance = (imprecise_predictions != -1)
    
    # calculate single-set length
    single_set_length = len(y_test) - sum(indeterminate_instance)
    
    # calculate determinacy
    determinacy = single_set_length/len(y_test)
    determinacy = round(determinacy*100, 2)
    
    # calculate single-set accuracy
    single_set_accuracy = sum(y_test[determinate_instance]==imprecise_predictions[determinate_instance])/single_set_length
    single_set_accuracy = round(single_set_accuracy*100, 2)
    
    # claculate u65
    u65_score = round(65 + (single_set_accuracy - 65)*determinacy/100, 2)
    
    # claculate precise accuracy
    precise_accuracy = sum(y_test==precise_predictions)/len(y_test)
    precise_accuracy = round(precise_accuracy*100, 2)
    
    # show confusion matrix
    if show_confusion_matrix:
        print('imprecise confusion matrix')
        cm1=metrics.confusion_matrix(y_test, imprecise_predictions)
        cm_display1=metrics.ConfusionMatrixDisplay(confusion_matrix=cm1,
                                        display_labels = [0, 1])
        cm_display1.plot()
    
        print('precise confusion matrix')
        cm2=metrics.confusion_matrix(y_test, precise_predictions)
        cm_display2=metrics.ConfusionMatrixDisplay(confusion_matrix=cm2,
                                        display_labels = [0, 1])
        cm_display2.plot()
    # return result
    return {'u65_score':u65_score, 
            'single_set_accuracy':single_set_accuracy, 
            'determinacy':determinacy, 
            'precise_accuracy':precise_accuracy,
            'abstention': 100-determinacy
            }

def random_forest_eval(X_test, y_test,model):
    return accuracy_score(y_test, model.predict(X_test))
