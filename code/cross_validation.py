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
from joblib import dump, load
stopw_list = myUtility.loadStopWlist()
num_features=0
mean_accuracy =0
mean_n_recall =0
mean_a_recall = 0
mean_r_recall = 0
mean_n_precision =0
mean_a_precision = 0
mean_r_precision = 0
mean_n_f1score =0
mean_a_f1score = 0
mean_r_f1score = 0
num_riduzioni=10 #indica il numero di volte di cui si devono ridurre le features
max_features=0
passo=0
algoritmo ='NB'

k_fold=10

myUtility.prepare_k_fold(k_fold)

#PRIMA PROVA SENZA FEATURE SELECTION
for round in range(0, k_fold):
	
	myUtility.prepare_test_train_set(k_fold, round+1)
	
	print('ROUND '+str(round))
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
	max_features = X_train_tfidf.shape[1]
	#k feature selection NO
	
	X_train_kf = X_train_tfidf
	
	#Training a classifier
	if algoritmo == 'NB':
		clf = MultinomialNB().fit(X_train_kf, train_s.target)
	else:
		clf = svm.SVC()
		clf.fit(X_train_kf, train_s.target)
	
	#TEST ===================================================================
	test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
	mylist=test_s.data
	testingData=[item.decode('cp437') for item in mylist]
	
	docs_new = testingData
	
	testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
	testingData_stemmed = myUtility.semming(testingData_noLinkENum)
	
	X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
	X_new_kf = X_new_tfidf
	
	predicted = clf.predict(X_new_kf)#prediction
	
	accuracy=np.mean(predicted == test_s.target)
	print("Accuracy on test set:")
	print(accuracy)
	f=open('accuracy.txt', "a")
	f.write(str(accuracy)+'\n')
	f.close()
	print("Metrics per class on test set:")
	print(metrics.classification_report(test_s.target, predicted,
		target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
	matr = metrics.confusion_matrix(test_s.target, predicted)
	print("Confusion matrix:")
	
	str_mat='NB - NO FEATURE SEL - ROUND: '+str(round)+'\n'
	for a in range(3):
		for b in range(3):
			str_mat=str_mat+str(matr[a][b]);
			if b<2: str_mat=str_mat+';'
		str_mat=str_mat+'\n'
	print(str_mat)			
	f=open('confusion_matrix.txt', "a")
	f.write(str_mat)
	f.close()
	print('\n')
# no fieat sel SVM============================================================================================
algoritmo ='svm'
for round in range(0, k_fold):
	
	myUtility.prepare_test_train_set(k_fold, round+1)
	
	print('ROUND '+str(round))
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
	max_features = X_train_tfidf.shape[1]
	#k feature selection NO
	
	X_train_kf = X_train_tfidf
	
	#Training a classifier
	if algoritmo == 'NB':
		clf = MultinomialNB().fit(X_train_kf, train_s.target)
	else:
		clf = svm.SVC()
		clf.fit(X_train_kf, train_s.target)
	
	#TEST ===================================================================
	test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
	mylist=test_s.data
	testingData=[item.decode('cp437') for item in mylist]
	
	docs_new = testingData
	
	testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
	testingData_stemmed = myUtility.semming(testingData_noLinkENum)
	
	X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
	X_new_kf = X_new_tfidf
	
	predicted = clf.predict(X_new_kf)#prediction
	
	accuracy=np.mean(predicted == test_s.target)
	print("Accuracy on test set:")
	print(accuracy)
	f=open('accuracy.txt', "a")
	f.write(str(accuracy)+'\n')
	f.close()
	print("Metrics per class on test set:")
	print(metrics.classification_report(test_s.target, predicted,
		target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
	matr = metrics.confusion_matrix(test_s.target, predicted)
	print("Confusion matrix:")
	
	str_mat='SVM - NO FEATURE SEL - ROUND: '+str(round)+'\n'
	for a in range(3):
		for b in range(3):
			str_mat=str_mat+str(matr[a][b]);
			if b<2: str_mat=str_mat+';'
		str_mat=str_mat+'\n'
	print(str_mat)			
	f=open('confusion_matrix.txt', "a")
	f.write(str_mat)
	f.close()
	print('\n')

#==========================================================================================================================
passo = int(max_features/num_riduzioni)
num_features = max_features
algoritmo ='NB'
for z in range (num_riduzioni-1):
	num_features = num_features - passo
	for round in range(0, k_fold):
		
		myUtility.prepare_test_train_set(k_fold, round+1)
		
		print('ROUND '+str(round))
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
		if algoritmo == 'NB':
			clf = MultinomialNB().fit(X_train_kf, train_s.target)
		else:
			clf = svm.SVC()
			clf.fit(X_train_kf, train_s.target)
		
		#TEST ===================================================================
		test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
		mylist=test_s.data
		testingData=[item.decode('cp437') for item in mylist]
		
		docs_new = testingData
		
		testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
		testingData_stemmed = myUtility.semming(testingData_noLinkENum)
		
		X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
		X_new_kf = k_feat.transform(X_new_tfidf)
		
		predicted = clf.predict(X_new_kf)#prediction
		
		accuracy=np.mean(predicted == test_s.target)
		print("Accuracy on test set:")
		print(accuracy)
		f=open('accuracy.txt', "a")
		f.write(str(accuracy)+'\n')
		f.close()
		print("Metrics per class on test set:")
		print(metrics.classification_report(test_s.target, predicted,
			target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
		f=open('stat.txt', "a")
		f.write(metrics.classification_report(test_s.target, predicted, target_names=test_s.target_names)+'\n\n')
		f.close()
		matr = metrics.confusion_matrix(test_s.target, predicted)
		print("Confusion matrix:")
		
		str_mat='NB - ROUND: '+str(round)+' - NUM_FEAT:; '+str(num_features)+'\n'
		for a in range(3):
			for b in range(3):
				str_mat=str_mat+str(matr[a][b]);
				if b<2: str_mat=str_mat+';'
			str_mat=str_mat+'\n'
		print(str_mat)			
		f=open('confusion_matrix.txt', "a")
		f.write(str_mat)
		f.close()
		print('\n')
		
		
passo = int(max_features/num_riduzioni)
num_features = max_features
algoritmo ='SVM'
for z in range (num_riduzioni-1):
	num_features = num_features - passo
	for round in range(0, k_fold):
		
		myUtility.prepare_test_train_set(k_fold, round+1)
		
		print('ROUND '+str(round))
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
		if algoritmo == 'NB':
			clf = MultinomialNB().fit(X_train_kf, train_s.target)
		else:
			clf = svm.SVC()
			clf.fit(X_train_kf, train_s.target)
		
		#TEST ===================================================================
		test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
		mylist=test_s.data
		testingData=[item.decode('cp437') for item in mylist]
		
		docs_new = testingData
		
		testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
		testingData_stemmed = myUtility.semming(testingData_noLinkENum)
		
		X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
		X_new_kf = k_feat.transform(X_new_tfidf)
		
		predicted = clf.predict(X_new_kf)#prediction
		
		accuracy=np.mean(predicted == test_s.target)
		print("Accuracy on test set:")
		print(accuracy)
		f=open('accuracy.txt', "a")
		f.write(str(accuracy)+'\n')
		f.close()
		print("Metrics per class on test set:")
		print(metrics.classification_report(test_s.target, predicted,
			target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
		f=open('stat.txt', "a")
		f.write(metrics.classification_report(test_s.target, predicted, target_names=test_s.target_names)+'\n\n')
		f.close()
		matr = metrics.confusion_matrix(test_s.target, predicted)
		print("Confusion matrix:")
		
		str_mat='SVM - ROUND: '+str(round)+' - NUM_FEAT:; '+str(num_features)+'\n'
		for a in range(3):
			for b in range(3):
				str_mat=str_mat+str(matr[a][b]);
				if b<2: str_mat=str_mat+';'
			str_mat=str_mat+'\n'
		print(str_mat)			
		f=open('confusion_matrix.txt', "a")
		f.write(str_mat)
		f.close()
		print('\n')
#==========================================================================================================================
'''

passo = 40#int(max_features/num_riduzioni)
num_features = 240#max_features
algoritmo='NB'
for z in range (5):
	num_features = num_features - passo
	for round in range(0, k_fold):
		
		myUtility.prepare_test_train_set(k_fold, round+1)
		
		print('ROUND '+str(round))
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
		if algoritmo == 'NB':
			clf = MultinomialNB().fit(X_train_kf, train_s.target)
		else:
			clf = svm.SVC()
			clf.fit(X_train_kf, train_s.target)
		
		#TEST ===================================================================
		test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
		mylist=test_s.data
		testingData=[item.decode('cp437') for item in mylist]
		
		docs_new = testingData
		
		testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
		testingData_stemmed = myUtility.semming(testingData_noLinkENum)
		
		X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
		X_new_kf = k_feat.transform(X_new_tfidf)
		
		predicted = clf.predict(X_new_kf)#prediction
		
		accuracy=np.mean(predicted == test_s.target)
		print("Accuracy on test set:")
		print(accuracy)
		f=open('accuracy.txt', "a")
		f.write(str(accuracy)+'\n')
		f.close()
		print("Metrics per class on test set:")
		print(metrics.classification_report(test_s.target, predicted,
			target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
		f=open('stat.txt', "a")
		f.write(metrics.classification_report(test_s.target, predicted, target_names=test_s.target_names)+'\n\n')
		f.close()
		matr = metrics.confusion_matrix(test_s.target, predicted)
		print("Confusion matrix:")
		
		str_mat='NB - ROUND: '+str(round)+' - NUM_FEAT:; '+str(num_features)+'\n'
		for a in range(3):
			for b in range(3):
				str_mat=str_mat+str(matr[a][b]);
				if b<2: str_mat=str_mat+';'
			str_mat=str_mat+'\n'
		print(str_mat)			
		f=open('confusion_matrix.txt', "a")
		f.write(str_mat)
		f.close()
		print('\n')

algoritmo ='svm'
passo = 40#int(max_features/num_riduzioni)
num_features = 240#max_features		
for z in range (5):
	num_features = num_features - passo
	for round in range(0, k_fold):
		
		myUtility.prepare_test_train_set(k_fold, round+1)
		
		print('ROUND '+str(round))
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
		if algoritmo == 'NB':
			clf = MultinomialNB().fit(X_train_kf, train_s.target)
		else:
			clf = svm.SVC()
			clf.fit(X_train_kf, train_s.target)
		
		#TEST ===================================================================
		test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
		mylist=test_s.data
		testingData=[item.decode('cp437') for item in mylist]
		
		docs_new = testingData
		
		testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
		testingData_stemmed = myUtility.semming(testingData_noLinkENum)
		
		X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
		X_new_kf = k_feat.transform(X_new_tfidf)
		
		predicted = clf.predict(X_new_kf)#prediction
		
		accuracy=np.mean(predicted == test_s.target)
		print("Accuracy on test set:")
		print(accuracy)
		f=open('accuracy.txt', "a")
		f.write(str(accuracy)+'\n')
		f.close()
		print("Metrics per class on test set:")
		print(metrics.classification_report(test_s.target, predicted,
			target_names=test_s.target_names))#metrics extractions (precision    recall  f1-score   support)
		f=open('stat.txt', "a")
		f.write(metrics.classification_report(test_s.target, predicted, target_names=test_s.target_names)+'\n\n')
		f.close()
		matr = metrics.confusion_matrix(test_s.target, predicted)
		print("Confusion matrix:")
		
		str_mat='SVM - ROUND: '+str(round)+' - NUM_FEAT:; '+str(num_features)+'\n'
		for a in range(3):
			for b in range(3):
				str_mat=str_mat+str(matr[a][b]);
				if b<2: str_mat=str_mat+';'
			str_mat=str_mat+'\n'
		print(str_mat)			
		f=open('confusion_matrix.txt', "a")
		f.write(str_mat)
		f.close()
		print('\n')
		
'''

'''
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
k_feat=SelectKBest(mutual_info_classif, k=400)
X_train_kf = k_feat.fit_transform(X_train_tfidf,train_s.target)

#Training a classifier
clf = MultinomialNB().fit(X_train_kf, train_s.target)

#TEST ===================================================================
test_s=load_files("data_k_fold/testset",description=None, load_content=True, shuffle=True, random_state=42)
mylist=test_s.data
testingData=[item.decode('cp437') for item in mylist]
print(test_s)
docs_new = testingData

testingData_noLinkENum = myUtility.tweetPreprocessing(testingData, stopw_list)
testingData_stemmed = myUtility.semming(testingData_noLinkENum)

X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
X_new_kf = k_feat.transform(X_new_tfidf)

predicted = clf.predict(X_new_kf)#prediction

accuracy=np.mean(predicted == test_s.target)
print("Accuracy on test set:")
print(accuracy)
print(predicted)


'''
