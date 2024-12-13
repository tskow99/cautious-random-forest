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
    return X_train, X_calib,X_test, y_train, y_calib, y_test

def heart_disease_data():
    HD=pd.read_csv('heart_disease_uci.csv')
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
    return X_train, X_calib,X_test, y_train, y_calib, y_test
