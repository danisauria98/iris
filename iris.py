# -*- coding: utf-8 -*-
"""IRIS

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qJqeTyMj20q-c6FVmkxFTM8GCJf91d7z

Universidad Autónoma de Chihuahua

Facultad de Ingeniería

Machine Learning

Tarea : IRIS Dataset

Profesora: M.A. Olanda Prieto

Ing. Daniela Alejandra Rubio Vega

MIC P372953
"""

from google.colab import drive
drive.mount('/content/drive')



#Importar librerias necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataset = load_iris()

##1.- Download dataset

#Import dataset
dataset = pd.read_csv('/content/drive/MyDrive/IRIS (1)/IRIS.csv')

dataset.info()

#Checar vals nulos
dataset.isnull().sum()

#Checar clases balanceadas
sns.countplot(x= 'species', data= dataset)

dataset.species.value_counts()

#Heat Map
sns.heatmap(data= dataset.corr(), annot= True)

dataset.head(10)

dataset.describe()

dataset.columns

dataset.shape

dataset['species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%'
                                        ,shadow=True, figsize=(10,8))
plt.show()

sns.FacetGrid(dataset, hue="species") \
   .map(plt.scatter, "sepal_length", "sepal_width") \
   .add_legend()

# Observar fts individuales a traves de una boxplot
ax = sns.boxplot(x="species", y="petal_length", data=dataset)
ax = sns.stripplot(x="species", y="petal_length", data=dataset, jitter=True, edgecolor="gray")

sns.pairplot(dataset, hue='species');

##2.- Train y Test Partition

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

x = dataset.drop('species', axis=1)
y= dataset.species
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

!pip install joblib

##3.- Crea un pipeline que realice la transformación de los datos
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib  # Importa joblib directamente

# Crear el pipeline con la transformación
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Paso de escalamiento usando StandardScaler
])

# Ajustar el pipeline a tus datos de entrenamiento
x_train_transf = pipeline.fit_transform(x_train)

x_train_transf[:10]

##4.- Guardar el pipeline en un archivo .sav
joblib.dump(pipeline, '/content/drive/MyDrive/IRIS (1)/pipeline_transformer.sav')

#Librerias para los modelos

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

#Logistic Regression model
logreg = LogisticRegression()
logreg.fit(x, y)
y_pred = logreg.predict(x)
print(metrics.accuracy_score(y, y_pred))

#SVM model
svm = SVC(kernel='rbf', random_state=42, gamma=.10, C=1.0)
svm.fit(x_train, y_train)

print('La accuracy del svm classifier con el training set is {:.2f} de  1'.format(svm.score(x_train, y_train)))

#DecisionTree Model

decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(x_train, y_train)

print('The accuracy of the Decision Tree classifier on training data is {:.2f} out of 1'.format(decision_tree.score(x_train, y_train)))

lr = LogisticRegression()
svm = SVC()
dt = DecisionTreeClassifier()

data = [(lr, [{'C': [0.01, 0.1, 0.5, 1.0], 'random_state':[42]}]),
        (svm, [{'C': [0.1, 0.5, 1.0], 'kernel': ['linear', 'rbf'], 'random_state':[42]}]),
        (dt, [{'criterion':['gini', 'entropy'], 'max_depth':[None, 10, 20, 30, 40, 50], 'min_samples_split':[2, 5],'min_samples_leaf': [1, 2, 4]}])]

from sklearn.model_selection import GridSearchCV

##5.- Mediante un GridSearch  o RandomizedSearchCV  busca los mejores parámetros para los 3 tipos de modelos (SVM, LR, Decision Trees
for i,j in data:
  grid = GridSearchCV(estimator = i , param_grid = j , scoring = 'accuracy',cv = 10)
  grid.fit(x_train,y_train)
  best_accuracy = grid.best_score_
  best_parameters = grid.best_params_
  print('{} \nBestAccuracy : {:.2f}%'.format(i,best_accuracy*100))
  print('BestParameters : ',best_parameters)
  print('')

##6.-Una vez que ya identificaste los mejores modelos, utiliza estos para realizar tus predicciones  con los datos de Test.  Si tus modelos fueron entrenados con datos transformados (eje. standard scale), será necesario que tu conjunto de datos de test sea transformado.


# Crear un StandardScaler para transformar los datos de prueba
scaler = StandardScaler()

# Ajustar el StandardScaler a los datos de entrenamiento y transformar los datos de prueba
x_test_scaled = scaler.fit_transform(x_test)

# Los mejores parámetros ya han sido encontrados en la búsqueda de hiperparámetros anterior

# Modelo de Logistic Regression con los mejores parámetros
best_lr = LogisticRegression(C=0.5, random_state=42)
best_lr.fit(x_train_transf, y_train)
y_pred_lr = best_lr.predict(x_test_scaled)

# Modelo de SVM con los mejores parámetros
best_svm = SVC(C=1.0, kernel='linear', random_state=42)
best_svm.fit(x_train_transf, y_train)
y_pred_svm = best_svm.predict(x_test_scaled)

# Modelo de Decision Tree con los mejores parámetros
best_dt = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                 min_samples_split=4,
                                 min_samples_leaf=2)
best_dt.fit(x_train_transf, y_train)
y_pred_dt = best_dt.predict(x_test_scaled)

##7.- Obtén su accuracy, recall y precision, F1 score, confusion matrix y reporte de clasificación (classification_report).
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report

# Evaluar Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr, average='weighted')
precision_lr = precision_score(y_test, y_pred_lr, average='weighted')
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
classification_report_lr = classification_report(y_test, y_pred_lr)

# Imprimir resultados
print("Logistic Regression:")
print("Accuracy:", accuracy_lr)
print("Recall:", recall_lr)
print("Precision:", precision_lr)
print("F1 Score:", f1_lr)
print("Confusion Matrix:")
print(confusion_matrix_lr)
print("Classification Report:")
print(classification_report_lr)

# Evaluar SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
classification_report_svm = classification_report(y_test, y_pred_svm)

# Imprimir resultados
print("\nSVM:")
print("Accuracy:", accuracy_svm)
print("Recall:", recall_svm)
print("Precision:", precision_svm)
print("F1 Score:", f1_svm)
print("Confusion Matrix:")
print(confusion_matrix_svm)
print("Classification Report:")
print(classification_report_svm)

# Evaluar Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)
classification_report_dt = classification_report(y_test, y_pred_dt)


# Imprimir resultados
print("\nDecision Tree:")
print("Accuracy:", accuracy_dt)
print("Recall:", recall_dt)
print("Precision:", precision_dt)
print("F1 Score:", f1_dt)
print("Confusion Matrix:")
print(confusion_matrix_dt)
print("Classification Report:")
print(classification_report_dt)

##8.- Almacena tus modelos entrenados en un archivo .sav.
# Nombre de los archivos .sav para guardar los modelos
lr_model_file = '/content/drive/MyDrive/IRIS (1)/lr_model.sav'
svm_model_file = '/content/drive/MyDrive/IRIS (1)/svm_model.sav'
dt_model_file = '/content/drive/MyDrive/IRIS (1)/dt_model.sav'

# Guardar los modelos entrenados en archivos .sav
joblib.dump(best_lr, lr_model_file)
joblib.dump(best_svm, svm_model_file)
joblib.dump(best_dt, dt_model_file)
