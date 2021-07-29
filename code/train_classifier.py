import myUtility
from sklearn import svm
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import re
import mysql.connector
from joblib import dump, load
# ----- PARAMETRI DI CONFIGURAZIONE -----
num_features = 1400 #numero di features
db_trainset_table ='trainset_table' #tabella contenente il trainset
name_classifier = 'NB' # NB or SVM
num_fold = 10 #per k fold
# ---------------------------------------

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="tweet_coronavirus"
)
myUtility.prepare_dataset(db_trainset_table)
stopw_list = myUtility.loadStopWlist()

# cross validation
accuracy_vec=list()

#decommentare per fare la 10-fold cross validation
'''
myUtility.prepare_k_fold(num_fold)
for round in range(0, num_fold):
	myUtility.prepare_test_train_set(num_fold, round+1)
	
	#converting binaries to strings
	train_s=load_files("data_k_fold/trainset",description=None, load_content=True, shuffle=True, random_state=42)
	mylist=train_s.data
	trainingData=[item.decode('cp437') for item in mylist]

	trainingData_noLinkENum = myUtility.tweetPreprocessing(trainingData, stopw_list)
	trainingData_stemmed = myUtility.semming(trainingData_noLinkENum)
	
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(raw_documents=trainingData_stemmed)
	
	# TF-IDF extraction
	tfidf_transformer = TfidfTransformer()# includes calculation of TFs (frequencies) and IDF
	X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
	#k feature selection
	k_feat=SelectKBest(mutual_info_classif, k=num_features)
	X_train_kf = k_feat.fit_transform(X_train_tfidf,train_s.target)
	
	#Training a classifier
	if name_classifier == 'NB':
		clf = MultinomialNB().fit(X_train_kf, train_s.target)
	else:
		clf = svm.SVC()
		clf.fit(X_train_kf, train_s.target)
	
	#TEST ===================================================================
	test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
	mylist=test_s.data
	testingData=[item.decode('cp437') for item in mylist]
	
	testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
	testingData_stemmed = myUtility.semming(testingData_noLinkENum)
	
	X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
	X_new_kf = k_feat.transform(X_new_tfidf)
	
	predicted = clf.predict(X_new_kf)#prediction
	
	accuracy_vec.append(np.mean(predicted == test_s.target))

	print("TEST "+str(round+1)+" - Metrics per class on test set:")
	print(metrics.classification_report(test_s.target, predicted,
		target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
		
	print('\n')
	
print('MEAN ACCURACY = '+str(np.mean(accuracy_vec)))
'''
# ADDESTRAMENTO CLASSIFICATORE
train_s=load_files("datasetTweet",description=None, load_content=True, shuffle=True, random_state=42)

mylist=train_s.data
trainingData=[item.decode('cp437') for item in mylist]
trainingData_noLinkENum = myUtility.tweetPreprocessing(trainingData, stopw_list)
trainingData_stemmed = myUtility.semming(trainingData_noLinkENum)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(raw_documents=trainingData_stemmed)
# TF-IDF extraction
tfidf_transformer = TfidfTransformer()# includes calculation of TFs (frequencies) and IDF
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#k feature selection
k_feat=SelectKBest(mutual_info_classif, k=num_features)
X_train_kf = k_feat.fit_transform(X_train_tfidf,train_s.target)
#Training a classifier
if name_classifier == 'NB':
	clf = MultinomialNB().fit(X_train_kf, train_s.target)
else:
	clf = svm.SVC()
	clf.fit(X_train_kf, train_s.target)

#SALVATAGGIO MODELLO
dump(clf, 'modello/modello_classificatore.joblib')
dump(count_vect, 'modello/count_vect.joblib')
dump(tfidf_transformer, 'modello/tfidf_transformer.joblib')
dump(k_feat, 'modello/k_feat.joblib')

print('THE CLASSIFIER HAS BEEN SAVED.') 