import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from ucimlrepo import fetch_ucirepo 

"""
Dataset Loader

This program provides an interface for loading and preprocessing various datasets. The datasets are preprocessed to be compatible with classification models and split into 
training, calibration, and testing subsets.

**Dataset Support**:
   - `german_credit_data`: German Credit Dataset from UCI ML repository.
   - `compas_data`: COMPAS dataset for recidivism prediction.
   - `heart_disease_data`: Heart Disease dataset from UCI.
   - `breast_cancer_data`: Breast Cancer Wisconsin Diagnostic dataset from UCI. 
"""

def load_dataset(dataset_name):

    if dataset_name == 'german_credit_data':
        return german_credit_data()
    elif dataset_name == 'compas_data':
        return compas_data()
    elif dataset_name == 'heart_disease_data':
        return heart_disease_data()
    elif dataset_name == 'breast_cancer_data':
        return breast_cancer_data()
    else:
        raise ValueError(f"Dataset: {dataset_name} not found")
def german_credit_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [
        "Status_of_existing_checking_account", "Duration_in_month", "Credit_history",
        "Purpose", "Credit_amount", "Savings_account_bonds", "Employment_since",
        "Installment_rate", "Personal_status_and_sex", "Other_debtors", "Present_residence_since",
        "Property", "Age_in_years", "Other_installment_plans", "Housing", "Number_of_existing_credits",
        "Job", "Number_of_people_liable", "Telephone", "Foreign_worker", "Target"
    ]

    data = pd.read_csv(url, delimiter=' ', names=columns, header=None)
    X = data.drop(columns=["Target"])
    y = data["Target"]
    y = y - 1
    categorical_features = X.select_dtypes(include=["object"]).columns

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    X = X.drop(columns=categorical_features)
    X = pd.concat([X.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

    X_train_calib, X_test, y_train_calib, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib, y_train_calib, test_size=0.1, random_state=42, stratify=y_train_calib)
    return {
        "train":X_train, 
        "calib": X_calib,
        "test": X_test,
        "train_labels": y_train,
        "calib_labels": y_calib,
        "test_labels": y_test
    }

def heart_disease_data():
    HD=pd.read_csv('./data/heart_disease_uci.csv')
    cols_to_drop=['id','dataset']
    HD_cleaned=HD.drop(columns=cols_to_drop)
    
    dummycols=['sex','cp','restecg','slope','thal']
    HD_cleaned=pd.get_dummies(HD_cleaned,columns=dummycols,drop_first=True)
    
    dropcols=['trestbps','fbs','oldpeak']
    HD_cleaned=HD_cleaned.drop(columns=dropcols)
    imp = IterativeImputer(max_iter=1000, random_state=0)
    imp=imp.fit(HD_cleaned)
    Ximp=imp.transform(HD_cleaned)
    Ximp=pd.DataFrame(Ximp,columns=HD_cleaned.columns)
    X=Ximp
    
    num_first=X.num
    num=np.where(num_first>0,1,0)
    X=X.drop(columns=['num'])
    kf=KFold(n_splits=10)
    X_train_calib, X_test, y_train_calib, y_test = train_test_split(X,
                                                      num,
                                                      test_size=0.33,
                                                      random_state=0)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib,
                                                          y_train_calib, 
                                                          test_size=0.1, 
                                                          random_state=0, 
                                                          stratify=y_train_calib)
    return {
        "train":X_train, 
        "calib": X_calib,
        "test": X_test,
        "train_labels": y_train,
        "calib_labels": y_calib,
        "test_labels": y_test
    }
def breast_cancer_data():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 
    y = y.Diagnosis == 'M'
    X_train_calib, X_test, y_train_calib, y_test = train_test_split(X,
                                                      y,
                                                      test_size=0.33,
                                                      random_state=0)
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib,
                                                          y_train_calib, 
                                                          test_size=0.1, 
                                                          random_state=0, 
                                                          stratify=y_train_calib)
    return {
        "train":X_train, 
        "calib": X_calib,
        "test": X_test,
        "train_labels": y_train,
        "calib_labels": y_calib,
        "test_labels": y_test
    }
def compas_data():
    df = pd.read_csv('./data/compas/compas_data_combined_matches.csv')
    columns_to_drop = ['FirstName', 'LastName', 'DateOfBirth', 'id', 'v_decile_score', 'DecileScore_Risk of Failure to Appear','race', 'DecileScore_Risk of Recidivism', 'DecileScore_Risk of Violence', 'RawScore_Risk of Failure to Appear', 'RawScore_Risk of Recidivism', 'RawScore_Risk of Violence', '_merge', 'sex', 'score_text', 'c_charge_desc']
    rf_dataset = df.drop(columns=columns_to_drop)
    ## Remove Nans
    na_counts = rf_dataset.isna().sum()
    na_columns = na_counts[na_counts > 0] 
    nans = na_columns.to_dict()
    columns_to_remove = []
    for key in nans.keys():
        columns_to_remove.append(key)
    rf_dataset = rf_dataset.drop(columns=columns_to_remove)
    labels = rf_dataset.two_year_recid
    rf_dataset = rf_dataset.drop(columns=['two_year_recid', 'is_recid'])

    X_train_calib, X_test, y_train_calib, y_test = train_test_split(rf_dataset, labels, test_size=0.2, random_state=42)
    
    X_train, X_calib, y_train, y_calib = train_test_split(X_train_calib, y_train_calib, test_size=0.2, random_state=42)
    return {
        "train":X_train, 
        "calib": X_calib,
        "test": X_test,
        "train_labels": y_train,
        "calib_labels": y_calib,
        "test_labels": y_test
    }