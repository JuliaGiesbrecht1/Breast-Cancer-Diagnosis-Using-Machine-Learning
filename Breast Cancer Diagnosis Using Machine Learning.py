#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Diagnosis Using Machine Learning

# By Julia Giesbrecht

# ### Introduction

# My focus for this project is on breast cancer diagnosis using machine learning. I will use the breast cancer Wisconsin (diagnostic) data set
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# This public dataset of breast cancer patients categorized tumors as either malignant or benign. It identifies features from a digitized image of a breast mass's fine needle aspirate (FNA). The attributes describe the characteristics of the cell nuclei present in the image.
# 
# How can machine learning improve breast cancer diagnosis accuracy and therefore reduce healthcare costs? (Reduce unnecessary treatments or time spent with each patient, improve workflow, etc.)

# ### Attribute Information

# 1) ID number
# 2) Diagnosis (M = malignant, B = benign)
# 3-32)
# 
# Ten real-valued features are computed for each cell nucleus:
#     a) radius (mean of distances from the center to points on the perimeter)
#     b) texture (standard deviation of gray-scale values)
#     c) perimeter
#     d) area
#     e) smoothness (local variation in radius lengths)
#     f) compactness (perimeter^2 / area - 1.0)
#     g) concavity (severity of concave portions of the contour)
#     h) concave points (number of concave portions of the contour)
#     i) symmetry
#     j) fractal dimension ("coastline approximation" - 1)
# 
# 
# Ten features were identified from each cell nucleus. Then, the mean, standard error, and "worst" or largest (mean of the three largest values) of these features were calculated, resulting in a total of 30 features. 
# 

# ### Objective and Motivation

# Breast cancer is one of the most common cancers among women worldwide. Unfortunately, it is often associated with few early symptoms, making early detection difficult and causing more challenges and uncertainty in treatments. However, early disease detection and diagnosis is imperative to reduce mortality and improve prognosis (Kalafi, 2019). Therefore, quickly and accurately diagnosing patients with breast cancer will allow patients to undergo life-saving treatments with a much higher chance of survivability. Furthermore, accurate classification of a benign tumor will reduce unnecessary healthcare costs and treatments (Milosevic et al., 2018). Using the breast cancer dataset, this project aims to accurately classify if the breast mass is benign or malignant to reduce healthcare costs and improve survival.
# 
# Milosevic, M., Jankovic, D., Milenkovic, A., & Stojanov, D. (2018, January 1). Early diagnosis and detection of breast cancer. Technology and Health Care. Retrieved November 8, 2022, from https://content.iospress.com/articles/technology-and-health-care/thc181277

# ### Import Libararies

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# ### Data Exploration

# In[3]:


# Read in data
data = pd.read_csv('data.csv')


# In[4]:


# First look at data
data.head()


# The first column is the ID column, the second is the target or diagnosis, and the last column is an unnamed file. The diagnosis column is separated into M = malignant, B = benign to showcase the tumors that are cancerous (malignant) or not cancerous (benign).

# In[5]:


# Looking at the shape of the data
data.shape


# The shape of the data shows 569 rows and 33 columns. The first column is the ID column, the second is the target or diagnosis and the last comlumn is an unnamed file. Therefore, there are 569 patients and 30 features in this dataset.

# In[6]:


# Counting number of benign(B) and malignant(M) cases
data['diagnosis'].value_counts()


# There are 357 Benign (B) cases, 212 malignant (M) cases.

# In[7]:


# Data visualization of benign(B) and malignant(M) cases
plt.figure(figsize=(8, 6))
sns.countplot(data['diagnosis'])
plt.xlabel("Diagnosis")
plt.title("Count Plot of Diagnosis")


# In[8]:


# checking for null values
data.isnull().sum()


# There are no null values in the data.

# In[9]:


# Checking data types
data.dtypes


# The diagnosis data type is an "object".

# In[10]:


# change "diagnosis" data type
labelencoder_Y = preprocessing.LabelEncoder()
data.iloc[:,1] = labelencoder_Y.fit_transform(data.iloc[:,1].values)


# I changed the data type of the "diagnosis" column to an integer. Now, 
# benign is (0), and malignant is (1).

# In[11]:


data.describe()


# ### Correlations

# In[12]:


corr_matrix = data.corr()
corr_matrix


# In[13]:


# threshold to explore highly correlated features
threshold = 0.75
filter_ = np.abs(corr_matrix['diagnosis']) > threshold
corr_features = corr_matrix.columns[filter_].tolist()
corr_features


# In[14]:


# visualize which features are highly correlated
sns.heatmap(data[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features")
plt.show()


# I used a heatmap to explore highly correlated features and set a threshold = 0.75 to find the features that were the most correlated. I did this to understand if there were correlated features that would need to be addressed later on using PCA.

# ### Create the Models

# To accurately classify the breast tumors, I used the KNN classifier and the Naive Bayes classifier. For the KNN classifier, I used grid search and for both models, I used PCA for model improvement. My goal for both models was to reduce the number of false negatives by finding the best parameters. I am doing this because not accurately classifying a patient with cancer could lead to death or impact life expectancy. It is more critical to flag someone as having cancer who does not have it than missing someone who does have cancer. Therefore, I am focusing first on the recall score and second the F1 score because the F1 score balances the weighting of both precision and recall.

# In[15]:


# splitting the data - features (X), diagnosis (Y)
X = data.iloc[:,2:32]
Y = data.iloc[:,1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature scaling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ### K-Nearest Neighbor Algorithm 

# In[16]:


# Grid search

knn = KNeighborsClassifier()
k_range = list(range(1, 21))
param_grid = dict(n_neighbors=k_range)
  
# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=5, scoring='recall', return_train_score=False,verbose=1)
  
# fitting the model for grid search
grid_search=grid.fit(X_train, Y_train)

print(grid_search.best_params_)
recall = grid_search.best_score_ *100
print("Recall for our training dataset with tuning is : {:.2f}%".format(recall))


# I used grid search to find the model's optimal parameters, which results in the most 'accurate' predictions. The results showed that using K=1 got the best results with the highest recall score.
# 

# In[20]:


# Nearest Neighbor algorithm 
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(X_train, Y_train)
Y_pred= classifier.predict(X_test)
Y_pred_train = classifier.predict(X_train)

# confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix:\n ", cm)

print('Accuracy Score : ' + str(accuracy_score(Y_test,Y_pred)))
print('Precision Score : ' + str(precision_score(Y_test,Y_pred)))
print('Recall Score : ' + str(recall_score(Y_test,Y_pred)))
print('F1 Score : ' + str(f1_score(Y_test,Y_pred)))
print('MSE Test: ' + str(mean_squared_error(Y_test,Y_pred)))
print('MSE Train: '+ str(mean_squared_error(Y_train,Y_pred_train)))


# I am using the nearest neighbor algorithm with the K = 1 because it produced the best recall results.
# 

# ### K-Nearest Neighbor Algorithm using PCA

# In[27]:


# Choosing the number of components with the best recall

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)


components = range(1,31)
best_rc = 0
best_c = None
best_p = None
best_f = None
best_ac = None
best_CM = None
best_mse = None
best_mse_t = None


for i in components:
   
    # Apply PCA
    pca = PCA(n_components = i) 
    x_scaled_pca = pca.fit_transform(x_scaled)
    
    # Splitting the dataset into the Training set and Test set
    X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(x_scaled_pca, Y, test_size = 0.25, random_state = 0)


    # Nearest Neighbor algorithm 
    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(X_train_pca, Y_train_pca)
    Y_pred_pca= classifier.predict(X_test_pca)
    
    ac_score =  accuracy_score(Y_test_pca,Y_pred_pca)
    p_score = precision_score(Y_test_pca,Y_pred_pca)
    re_score =  recall_score(Y_test_pca,Y_pred_pca)
    f_Score = f1_score(Y_test_pca,Y_pred_pca)
    cm = confusion_matrix(Y_test_pca, Y_pred_pca)
    mse = mean_squared_error(Y_test,Y_pred)
    mset = mean_squared_error(Y_train,Y_pred_train)

    
    if re_score > best_rc:
        best_rc = re_score
        best_c = i
        best_p = p_score
        best_f = f_Score
        best_ac = ac_score
        best_CM = cm
        best_mse = mse
        best_mse_t = mset
    
        
                    
#     print(f'Recall score is with {i} components and the score is:  {best_rc}') 
#     print(mse)
#     print(mset)
print(f'Confusion matrix: \n {cm}')       
print(f'The best recall score is with {best_c} components and the score is:  {best_rc}')
print(f'Accuracy score: {best_ac}')
print(f'Precision score: {best_p}')
print(f'Recall score: {best_rc}')
print(f'f1 score: {best_f}')
print(f'MSE Test: {best_mse}')
print(f'MSE Train: {best_mse_t}')


# To further increase the recall and F1 score and address the multicollinearity, I used PCA with the KNN classifier and found that PCA with four components got the best results.

# ### Naïve Bayes Algorithm

# Next, I used the same train and test set to classify the data for the Naive Bayes classifier. I then used PCA with the Naive Bayes classifier to improve the model and address multicollinearity. Again, I found the best recall score using one component

# In[34]:


# Naïve Bayes Algorithm
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
Y_pred= classifier.predict(X_test)


# confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix:\n ", cm)

print('Accuracy Score : ' + str(accuracy_score(Y_test,Y_pred)))
print('Precision Score : ' + str(precision_score(Y_test,Y_pred)))
print('Recall Score : ' + str(recall_score(Y_test,Y_pred)))
print('F1 Score : ' + str(f1_score(Y_test,Y_pred)))
print('MSE Test: ' + str(mean_squared_error(Y_test,Y_pred)))
print('MSE Train: '+ str(mean_squared_error(Y_train,Y_pred_train)))


# ### Naïve Bayes Algorithm with PCA

# In[46]:


# Choosing the number of components

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)


components = range(1,31)
best_rc = 0
best_c = None
best_p = None
best_f = None
best_ac = None
best_CM = None
best_mse = None
best_mse_t = None

for i in components:
   
    # Apply PCA
    pca = PCA(n_components = i) 
    x_scaled_pca = pca.fit_transform(x_scaled)
    
    # Splitting the dataset into the Training set and Test set
    X_train_pca, X_test_pca, Y_train_pca, Y_test_pca = train_test_split(x_scaled_pca, Y, test_size = 0.25, random_state = 0)


    # Naïve Bayes Algorithm
    classifier = GaussianNB()
    classifier.fit(X_train_pca, Y_train_pca)
    Y_pred_pca= classifier.predict(X_test_pca)
    
    ac_score =  accuracy_score(Y_test_pca,Y_pred_pca)
    p_score = precision_score(Y_test_pca,Y_pred_pca)
    re_score =  recall_score(Y_test_pca,Y_pred_pca)
    f_Score = f1_score(Y_test_pca,Y_pred_pca)
    cm = confusion_matrix(Y_test_pca, Y_pred_pca)
    mse = mean_squared_error(Y_test,Y_pred)
    mset = mean_squared_error(Y_train,Y_pred_train)

    
    if re_score > best_rc:
        best_rc = re_score
        best_c = i
        best_p = p_score
        best_f = f_Score
        best_ac = ac_score
        best_CM = cm
        best_mse = mse
        best_mse_t = mset
                    
    #print(f'Accuracy score is with {i} components and the score is:  {best_ac}')   
        
print(f'The best recall score is with {best_c} components and the score is:  {best_rc}')
print(f'Accuracy score: {best_ac}')
print(f'Precision score: {best_p}')
print(f'Recall score: {best_rc}')
print(f'f1 score: {best_f}')
print(f'Confusion matrix: \n {cm}')
print(f'MSE Test: {best_mse}')
print(f'MSE Train: {best_mse_t}')


# To test for overfitting, I compared the mean square error (MSE) for the test data vs the training data. I found in all models that the MSE for the training error was zero indicating overfitting. However, I did not change model parameters to address overfitting because I got optimal results from the test data with a high recall, F1 and accuracy scores and a low MSE score.

# ## Results

# 
# ![image.png](attachment:image.png)
# 

# KNN using PCA had the best recall score, which means it has the least number of people being falsely diagnosed as benign or slipping through the cracks of the medical system. However, KNN, without the use of PCA has the best F1 score and a high but slightly lower recall. This means more people are not diagnosed with cancer when they should have been. However, it may be better from a business standpoint because it has a better balance of false negatives and false positives, as represented by the F1 score. Moreover, KNN without PCA has a higher precision score. Therefore, fewer people are being falsely diagnosed as having cancer, which means less tests are being done and less money is being spent. Therefore, KNN without using PCA is the best model from a business lens.
# 
# Interestingly, the Naive Bayes algorithm had the lowest scores overall compared to the KNN scores. The KNN algorithm significantly outperformed the Naive Bayes algorithm.

# ## Conclusion

# To address the business question on how to use machine learning to best predict breast cancer and therefore reduce healthcare costs, the best model balances both false negative and false positive scores as represented by F1 score. Therefore, KNN without using PCA is the best model because it has the second highest recall and the best F1 score, therefore, it accurately diagnoses the greatest number of people while also reducing unnecessary healthcare costs.

# In[ ]:




