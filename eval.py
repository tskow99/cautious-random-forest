from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score



def evaluate_model(model_name, X_test, y_test,model):
    if model_name == 'ConformalPredictor':
        return conformal_pred_eval(X_test, y_test,model)
    elif model_name == 'WCRF':
        return wcrf_eval(X_test, y_test,model)
    elif model_name == 'vanillaRF':
       return vanilla_rf_eval(X_test, y_test,model) 
    else:
        raise ValueError(f"Dataset: {model_name} not found")

     

def conformal_pred_eval(X_test, y_test,model):
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
            'determinacy':determinacy}


def wcrf_eval(X_test, y_test,model,  plot=False, show_confusion_matrix=False):
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
            'precise_accuracy':precise_accuracy}

def vanilla_rf_eval(X_test, y_test,model):
    return accuracy_score(y_test, model.predict(X_test))