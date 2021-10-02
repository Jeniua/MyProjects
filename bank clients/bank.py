import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

df = pd.read_csv('/datasets/Churn.csv')
display(df.head())

df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
#print(df['Tenure'].mean().value_counts())
print(df['Tenure'].value_counts())

gender_OHE = pd.get_dummies(df["Gender"], drop_first=True)
geography_OHE = pd.get_dummies(df['Geography'], drop_first = True)
df.drop(['Gender', 'Geography'],axis = 1, inplace = True)
df_OHE = pd.concat([df, gender_OHE, geography_OHE],axis = 1)

df_OHE.drop(['RowNumber', "CustomerId", "Surname"], axis=1, inplace=True)

#Посмотрим на баланс классов.

print(len(df_OHE.query('Exited == 1'))/len(df_OHE), 'Доля положительных классов')
print(len(df_OHE.query('Exited != 1'))/len(df_OHE), 'Доля отрицательных классов')

features = df_OHE.drop(['Exited'], axis=1)
target = df_OHE['Exited']

# разделим массив на 3 части: 60% - обучающая, валидационная и тестовая по 20% 
# разбиваем массив на 2 части - тестовую и остальное
features_train, features_other, target_train, target_other = train_test_split(features, target, test_size=0.40, random_state=12345)

# разбиваем остальное на 2 части - валидационную и тестовую
features_valid, features_test, target_valid, target_test = train_test_split(features_other, target_other, test_size=0.50, random_state=12345)
print('  обучающая -', len(features_train))
print('  валидирующая -', len(features_valid))
print('  тестовая -', len(features_test))


numeric = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

scaler = StandardScaler()
scaler.fit(features_train[numeric]) 
features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])

best_score = 0
for depth in range(1,10):
    model_1 = DecisionTreeClassifier(random_state = 12345,max_depth = depth)
    model_1.fit(features_train, target_train)
    predictions_valid = model_1.predict(features_valid)
    accuracy = accuracy_score(predictions_valid,target_valid)
    if accuracy > best_score:
        best_score = accuracy
        best_depth = depth
print('Дерево решенй')
print("max_depth =", best_depth, ": ",best_score)
print("F1:", round(f1_score(predictions_valid, target_valid),2))


model_2 = RandomForestClassifier(random_state = 12345, n_estimators = 40, max_depth = 9)
model_2.fit(features_train, target_train)

predictions = model_2.predict(features_valid)
print("accuracy:", accuracy_score(predictions, target_valid))
print("F1:", round(f1_score(predictions, target_valid), 2))

model_3 = LogisticRegression()
model_3.fit(features_train, target_train)
predictions_regr = model_3.predict(features_valid)
print("Accuracy:", accuracy_score(predictions_regr, target_valid))
print("F1:", f1_score(predictions_regr, target_valid))


# Разберемся с дисбалансом классов
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    target_zeros = target[target == 0]
    features_ones = features[target == 1]
    target_ones = target[target == 1]
    
    features_upsampled = pd.concat([features_zeros] + [features_ones]*repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones]*repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled
features_upsampled, target_upsampled = upsample(features_train, target_train, 10)

#model_2_upsample = GridSearchCV(model_2, param_grid, cv=5)
model_2.fit(features_upsampled, target_upsampled)
predictions_upsample = model_2.predict(features_valid)
print("accuracy:", accuracy_score(predictions_upsample, target_valid))
print("F1:", f1_score(predictions_upsample, target_valid), 2)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345)
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, fraction=0.4)

#CV_model_2_downsample = GridSearchCV(model_2, param_grid, cv=5)
model_2_downsample = model_2
model_2_downsample.fit(features_downsampled, target_downsampled)
predictions_downsample = model_2_downsample.predict(features_valid)
print("accuracy:", accuracy_score(predictions_downsample, target_valid))
print("F1:", round(f1_score(predictions_downsample, target_valid), 2))

predictions_test = model_2_downsample.predict(features_test)
print("accuracy:", accuracy_score(predictions_test, target_test))
print("F1:", round(f1_score(predictions_test, target_test), 2))

probabilities_test = model_2_downsample.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]
auc_roc = roc_auc_score(target_test,probabilities_one_test)

print('AUC-ROC: ',auc_roc)

#Имеющиеся данные разбили на три выборки, обучили три модели из них выбрали лучшую, ей оказалась модель случайного леса. 
#Провели баланс классов, улучшить значение f1 меры погло уменьшение выборки, проверили конечную модель на тестовой выборке, 
#получили значение F1: 0.6, AUC-ROC: 0.8519707797220986.
