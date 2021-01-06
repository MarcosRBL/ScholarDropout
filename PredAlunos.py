import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as m
import streamlit as st

import inflection
import pylab
import random
import warnings
import os
import io

from IPython.display import Image
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

#Streamlit page
st.set_page_config(page_title='Propensão de Evasão Escolar')

multiple_files = st.file_uploader(
    "Insira o arquivo CSV abaixo",
    accept_multiple_files=True
)
for file in multiple_files:
    file_container = st.beta_expander(
        f"Nome do Arquivo: {file.name} ({file.size})"
    )
    data = io.BytesIO(file.getbuffer())

    st.text("")

    if st.button('Estudar dados'):
        #Lugar onde todas as funções serão feitas
        def ml_error(model_name, y, yhat):
            print(model_name)
            print(classification_report(y_test, y_pred))

        #importando o dataset
        df_raw = pd.read_csv(data)

        #Em todas as novas seções vamos fazer uma cópia para caso alguma coisa não saia como o esperado podemos rodar apenas a seção 
        #novamente não necessitando rodar todo o projeto.
        df1 = df_raw.copy()

        cols_old = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
            'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
            'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
            'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
            'Walc', 'health', 'absences', 'passed']

        snackecase = lambda x: inflection.underscore(x)
        cols_new = list(map( snackecase, cols_old ) )

        df1.columns = cols_new

        num_attributes = df1.select_dtypes( include=['int64', 'float64'] )
        cat_attributes = df1.select_dtypes( exclude=['int64', 'float64'] )

        mms = MinMaxScaler()
        df1['age'] = mms.fit_transform( df1[['age']].values )
        df1['medu'] = mms.fit_transform( df1[['medu']].values )
        df1['fedu'] = mms.fit_transform( df1[['fedu']].values )
        df1['traveltime'] = mms.fit_transform( df1[['traveltime']].values )
        df1['studytime'] = mms.fit_transform( df1[['studytime']].values )
        df1['failures'] = mms.fit_transform( df1[['failures']].values )
        df1['famrel'] = mms.fit_transform( df1[['famrel']].values )
        df1['freetime'] = mms.fit_transform( df1[['freetime']].values )
        df1['goout'] = mms.fit_transform( df1[['goout']].values )
        df1['dalc'] = mms.fit_transform( df1[['dalc']].values )
        df1['walc'] = mms.fit_transform( df1[['walc']].values )
        df1['health'] = mms.fit_transform( df1[['health']].values )
        df1['absences'] = mms.fit_transform( df1[['absences']].values )

        le = LabelEncoder()
        df1['school'] = le.fit_transform( df1['school'] )
        df1['sex'] = le.fit_transform( df1['sex'] )
        df1['address'] = le.fit_transform( df1['address'] )
        df1['famsize'] = le.fit_transform( df1['famsize'] )
        df1['mjob'] = le.fit_transform( df1['mjob'] )
        df1['fjob'] = le.fit_transform( df1['fjob'] )
        df1['reason'] = le.fit_transform( df1['reason'] )
        df1['guardian'] = le.fit_transform( df1['guardian'] )
        df1['schoolsup'] = le.fit_transform( df1['schoolsup'] )
        df1['famsup'] = le.fit_transform( df1['famsup'] )
        df1['paid'] = le.fit_transform( df1['paid'] )
        df1['activities'] = le.fit_transform( df1['activities'] )
        df1['nursery'] = le.fit_transform( df1['nursery'] )
        df1['higher'] = le.fit_transform( df1['higher'] )
        df1['internet'] = le.fit_transform( df1['internet'] )
        df1['romantic'] = le.fit_transform( df1['romantic'])
        df1['passed'] = le.fit_transform( df1['passed'])
        df1['pstatus'] = le.fit_transform( df1['pstatus'])

        X = df1.drop('passed', axis=1)
        y = df1['passed']

        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.20 )

        rfc = RandomForestClassifier(random_state = 42)
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)

        from sklearn.model_selection import cross_val_score

        scores_dt = cross_val_score(rfc, X, y,
                                    scoring='accuracy', cv=5)

        accScore = 'Acurácia',accuracy_score(y_test, y_pred)* 100
        classRep = classification_report(y_test, y_pred)

        st.text(accScore)
        st.text(classRep)