# Titanic_proj  

* The dataset is from [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic/data).
* Created models(Random Forest Classifier, Decision Tree Classifier, KNN, Naive Bayes Classifier, SVM, XGboost) to predict the survival of passengers.
* Compare the prediction from [Extensive_EDA_plus_top2%](https://www.kaggle.com/boss0ayush/extensive-eda-plus-top2) with mine and reach the best model.

## Code and Resources Used

**Dataset from :** <https://www.kaggle.com/c/titanic/data>  
**Python Version :** Python 3.9.4  
**IDE :** VSCode, Spyder, Jupyter Notebook  
**Packages :** pandas, numpy, missingno, matplotlib, seaborn, plotly, sklearn, xgboost  
**Reference websites :**

1. [Extensive_EDA_plus_top2%](https://www.kaggle.com/boss0ayush/extensive-eda-plus-top2)  
2. [Random Forest on Titanic Dataset ⛵.](https://medium.com/analytics-vidhya/random-forest-on-titanic-dataset-88327a014b4d)  
3. [【進階分析】鐵達尼生存預測 - 手把手資料分析實戰教學](https://aifreeblog.herokuapp.com/posts/64/Data_Analytics_in_Practice_Titanic/)  
4. [Predicting the Survival of Titanic Passengers](https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8)  
5. [Kaggle – Titanic倖存預測 #1](https://chtseng.wordpress.com/2017/12/24/kaggle-titanic%E5%80%96%E5%AD%98%E9%A0%90%E6%B8%AC-1/)  
6. [使用GridSearch對xgboost進行調參（全部流程）](https://blog.csdn.net/jh1137921986/article/details/84754868)  

## Data info

After combining the train and the test dataset, we got 1309 records and 13 columns:
| Columns  | Definition  | Key |
| :------------ |:---------------:| -----:|
| PassengerId | Passenger ID |  |
| Survived | Survival | 0 = No, 1 = Yes |
| Pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
| Name | Name |  |
| Sex | Sex |  |
| Age | Age |  |
| SibSp | # of siblings / spouses aboard the Titanic |  |
| Parch | # of parents / children aboard the Titanic |  |
| Ticket | Ticket number |  |
| Fare | Passenger fare |  |
| Cabin | Cabin number |  |
| Embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
| train_test | Differentiate train and test dataset | 0 = test dataset, 1 = train dataset |

## Data Cleaning

Clean the data up to prepare for our models. I made the following changes:

* Use median to fill the missing values in 'Age'.
* Use median to fill the missing values in 'Fare'.  
* Use appear most frequently place to fill the missing values in 'Embarked'.

**Before Cleaning :**  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/before_cleaning.png" width=50%, heigh=50%>  

**After Cleaning :**  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/after_cleaning.png" width=50%, heigh=50%>

## Exploratory Data Analysis (EDA)

Below are a few tables and graphs I made. Try to analyzing and visualizing the dataset.
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Age_range%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Age_range_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Cabin_count%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Cabin_count_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Embarked%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Embarked_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Family_count%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Family_count_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Fare_range%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Fare_range_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Pclass%20vs%20Survived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Pclass_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Sex%20vs%20Pclass%20vs%20Survived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Sex_Pclass_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Sex%20vs%20Survived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Sex_Survived_rate.png" width=15%, heigh=30%>  
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/new_cabin%20vs%20Suvived%3D1.png" width=70%, heigh=65%>
<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/new_cabin_Survived_rate.png" width=15%, heigh=30%>  

## Model Building

1. I used 6 different models and evaluated them using ```cross_val_score``` .  

    | Model  | cross_val_score  |
    | :------------ |---------------:|
    | **Random Forest Classifier** | 0.810 |
    | **Decision Tree Classifier** | 0.804 |
    | **KNN** | ***0.820*** |
    | **Naive Bayes Classifier** | 0.711 |
    | **SVC** | 0.809 |
    | **XGboost** | 0.818 |

2. I also used GridsearchCV to find out the best group of parameters that can optimize these models.

    | Model  | best_score_  |
    | :------------ |---------------:|
    | **Random Forest Classifier** | 0.818 |
    | **Decision Tree Classifier** | 0.810 |
    | **KNN** | 0.820 |
    | **Naive Bayes Classifier** | 0.770 |
    | **SVC** | 0.815 |
    | **XGboost** | ***0.835*** |

## Model Performance

I used the prediction from [Extensive_EDA_plus_top2%](https://www.kaggle.com/boss0ayush/extensive-eda-plus-top2) to compare with mine.  
The **SVC model** has the best performance.  

| Model  | MAE  |
| :------------ |---------------:|
| **Random Forest Classifier** | 0.222 |
| **Decision Tree Classifier** | 0.246 |
| **KNN** | 0.279 |
| **Naive Bayes Classifier** | 0.282 |
| **SVC** | ***0.217*** |
| **XGboost** | 0.232 |

## Kaggle Competition

<img src="https://github.com/JohnnyHsieh1020/Titanic_proj/blob/main/images/Kaggle%20competition%20result.png" width=70%, heigh=65%>
