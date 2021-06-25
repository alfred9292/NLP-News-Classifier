import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_validate,cross_val_predict

count = 0

vectorizer = CountVectorizer()

#please change the path of file here#
df = pd.read_csv("/Users/alfred/PycharmProjects/nlp/topic.csv",delimiter=',', skiprows=1, header=None)

#split the data into training set and test set, intend to divide all dataset to training set to faciliate the 10 kold cross validation at later stage#
X_train_raw,X_test_raw,y_train,y_test = train_test_split(df[2],df[0],test_size=0,random_state=0)
print("train:",len(X_train_raw),"test:",len(X_test_raw))

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#cross validation with 10-fold,every interation it select 90% of the total dataset for training and 10% for testing#

accuary_scores = cross_validate(classifier, X_train, y_train, cv=10)
precision_scores = cross_validate(classifier, X_train, y_train, cv=10,scoring='precision_macro')
recall_scores = cross_validate(classifier, X_train, y_train, cv=10,scoring='recall_macro')
f1_scores = cross_validate(classifier, X_train, y_train, cv=10,scoring='f1_macro')
predicted = cross_val_predict(classifier,X_train,y_train,cv=10)


print('accuracy:',accuary_scores)
print('precision:',precision_scores)
print('recall:',recall_scores)
print('f1_scores:',f1_scores)

print('confusion matrix:')
print(confusion_matrix(y_train,predicted))
print('classfication report,')
print(classification_report(y_train,predicted))

#to pull out some instances based on predicted category to investigate correct and incorrect case for analysis purpose#
#only use when necessary#

#for i, predicted in enumerate(predicted):

    #if ' Surveillance Flights Revealed' in X_train_raw.iloc[i]:
     #count=count+1

     #print('predicted category:%s.title:%s'%(predicted,X_train_raw.iloc[i]))

#print('total count:',count)



