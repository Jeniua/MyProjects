import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.dummy import DummyClassifier



simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('/datasets/users_behavior.csv')

print(len(df))
print(df.info())
print(df.head(20))

# ## Разбейте данные на выборки <a name="step2"></a>



features = df.drop(['is_ultra'], axis = 1)
target = df['is_ultra']
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

features_train = train.drop(['is_ultra'], axis = 1)
target_train = train['is_ultra']

features_valid = validate.drop(['is_ultra'], axis = 1)
target_valid = validate['is_ultra']

features_test = test.drop(['is_ultra'], axis = 1)
target_test = test['is_ultra']

print(train.size,validate.size, test.size)


# <span style="color:blue">*Добавил размеры выборок для проверки.*</span>

# ## Исследуйте модели

# Начнём с модели дерева решений.


for depth in range(1, 7):
    best_score = 0
    model_1 = DecisionTreeClassifier(random_state = 12345, max_depth = depth)
    model_1.fit(features_train, target_train)
    predictions_valid = model_1.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions_valid)
    if accuracy > best_score:
        best_score = accuracy
        best_depth = depth
print('Дерево решенй')
print("max_depth =", best_depth, ": ",best_score)

# Теперь возьмём модель случайного леса



for est in range(1, 11):
    best_est = 0
    best_result = 0
    model_2 = RandomForestClassifier(random_state=12345, n_estimators=est) 
    model_2.fit(features_train, target_train) 
    result = model_2.score(features_valid, target_valid) 
    if result > best_result:
        best_est = est
        best_result = result
print('Случайный лес')
print(best_result, best_est)



model_3 = LogisticRegression(random_state = 12345)
model_3.fit(features_train, target_train)
result = model_3.score(features_valid, target_valid)
print('Логистическая регрессия')
print(result)




predictions_test = model_1.predict(features_test)
accuracy_test = accuracy_score(target_test, predictions_test)
print(accuracy_test)


# ## (бонус) Проверьте модели на адекватность

# <span style="color:blue">*Все что смог найти по поводу адекватности модели.*</span>



dummy = DummyClassifier(strategy='most_frequent').fit(features_test, target_test)
dummy_pred = dummy.predict(features_test)

print('Test score: ', accuracy_score(target_test, dummy_pred))
