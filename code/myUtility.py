from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
import numpy as np
import re
import shutil
import os
from os import listdir
import mysql.connector

#Carica stop words list
def loadStopWlist():
	stopw_list = list()
	file = open("stop_words.txt", "r") 
	line=file.readline()
	while line: 
		x = line.split()
		stopw_list.append(x[0])
		line=file.readline()
	file.close()
	return stopw_list

#rimuovere link  numeri hashtag e common words	
def tweetPreprocessing(trainingData, stopw_list):
	trainingData_noLinkENum=[]
	for testo in trainingData:
		test= testo.lower()
		s = re.split("\s+|'+", test)
		b=""
		for a in s:
			a = re.sub('[^0-9a-zA-Z]+', '', a)
			if re.search("http", a): 
				a="http"
			if re.match(r'^([\s\d]+)$', a):
				a=""
			if re.match(r'^#', a):
				a=re.sub("#", "", a)
			for sw in stopw_list:
				if re.match(" "+sw+" ", " "+a+" "):
					a=""
					break
			if a.isspace()==False:		
				b=b+a+" "
		if  b.isspace()==False:
			trainingData_noLinkENum.append(b)
	return trainingData_noLinkENum

def semming(trainingData_noLinkENum):
	stemmer = SnowballStemmer("italian")
	trainingData_stemmed=[]
	for testo in trainingData_noLinkENum:
		splittato = re.split("\s+|'+", testo)
		stemmato=[]
		for ts in splittato:
			stemmato.append(stemmer.stem(ts))
		
		a=''
		for ts in stemmato:
			if  ts.isspace()==False:
				a=a+ts+' '
		
		if  a.isspace()==False:
			trainingData_stemmed.append(a)
	return trainingData_stemmed
	
def prepare_k_fold(k):
	list_allarm = listdir('datasetTweet/allarmisti')
	list_neutri = listdir('datasetTweet/neutri')
	list_rassic = listdir('datasetTweet/rassicuranti')
	
	num_allarm= len(list_allarm)
	num_rassic= len(list_rassic)
	num_neutri= len(list_neutri)
	
	for i in range(1, 11):
		shutil.rmtree('data_k_fold/fold'+str(i)+'/neutri')
		shutil.rmtree('data_k_fold/fold'+str(i)+'/allarmisti')
		shutil.rmtree('data_k_fold/fold'+str(i)+'/rassicuranti')
		os.mkdir('data_k_fold/fold'+str(i)+'/neutri')
		os.mkdir('data_k_fold/fold'+str(i)+'/allarmisti')
		os.mkdir('data_k_fold/fold'+str(i)+'/rassicuranti')
	
	for i in range(0, k):
		lowbound=i*(int(num_allarm/k))
		upbound = lowbound+(int(num_allarm/k))
		source = 'datasetTweet/allarmisti/'
		destination= 'data_k_fold/fold'+str(i+1)+'/allarmisti'
		for j in range(lowbound, upbound):
			s= source+'/'+list_allarm[j]
			shutil.copy(s, destination)
			
		lowbound=i*(int(num_rassic/k))
		upbound = lowbound+(int(num_rassic/k))
		source = 'datasetTweet/rassicuranti'
		destination= 'data_k_fold/fold'+str(i+1)+'/rassicuranti'
		for j in range(lowbound, upbound):
			s= source+'/'+list_rassic[j]
			shutil.copy(s, destination)

		lowbound=i*(int(num_neutri/k))
		upbound = lowbound+(int(num_neutri/k))
		source = 'datasetTweet/neutri'
		destination= 'data_k_fold/fold'+str(i+1)+'/neutri'
		for j in range(lowbound, upbound):
			s= source+'/'+list_neutri[j]
			shutil.copy(s, destination)


def prepare_test_train_set(k,i):
	#svuotare test e train set
	
	shutil.rmtree('data_k_fold/testset/neutri')
	shutil.rmtree('data_k_fold/testset/allarmisti')
	shutil.rmtree('data_k_fold/testset/rassicuranti')
	shutil.rmtree('data_k_fold/trainset/neutri')
	shutil.rmtree('data_k_fold/trainset/allarmisti')
	shutil.rmtree('data_k_fold/trainset/rassicuranti')
	
	os.mkdir('data_k_fold/testset/neutri')
	os.mkdir('data_k_fold/testset/allarmisti')
	os.mkdir('data_k_fold/testset/rassicuranti')
	
	os.mkdir('data_k_fold/trainset/neutri')
	os.mkdir('data_k_fold/trainset/allarmisti')
	os.mkdir('data_k_fold/trainset/rassicuranti')
	
	for j in range(0, k):
		if j==(i-1):
			continue
		
		list_allarm = listdir('data_k_fold/fold'+str(j+1)+'/allarmisti')
		list_neutri = listdir('data_k_fold/fold'+str(j+1)+'/neutri')
		list_rassic = listdir('data_k_fold/fold'+str(j+1)+'/rassicuranti')
		
		for l in list_allarm:
			shutil.copy('data_k_fold/fold'+str(j+1)+'/allarmisti/'+l, 'data_k_fold/trainset/allarmisti')

		for l in list_neutri:
			shutil.copy('data_k_fold/fold'+str(j+1)+'/neutri/'+l, 'data_k_fold/trainset/neutri')

		for l in list_rassic:
			shutil.copy('data_k_fold/fold'+str(j+1)+'/rassicuranti/'+l, 'data_k_fold/trainset/rassicuranti')

	list_allarm = listdir('data_k_fold/fold'+str(i)+'/allarmisti')
	list_neutri = listdir('data_k_fold/fold'+str(i)+'/neutri')
	list_rassic = listdir('data_k_fold/fold'+str(i)+'/rassicuranti')
	
	for l in list_allarm:
		shutil.copy('data_k_fold/fold'+str(i)+'/allarmisti/'+l, 'data_k_fold/testset/allarmisti')

	for l in list_neutri:
		shutil.copy('data_k_fold/fold'+str(i)+'/neutri/'+l, 'data_k_fold/testset/neutri')

	for l in list_rassic:
		shutil.copy('data_k_fold/fold'+str(i)+'/rassicuranti/'+l, 'data_k_fold/testset/rassicuranti')


def prepare_dataset(tablename):
	mydb = mysql.connector.connect(
		host="localhost",
		user="root",
		passwd="",
		database="tweet_coronavirus"
	)
	mycursor = mydb.cursor()
	sql = "SELECT text, etichetta FROM "+tablename
	mycursor.execute(sql)
	result= mycursor.fetchall()
	
	shutil.rmtree('datasetTweet/neutri')
	shutil.rmtree('datasetTweet/allarmisti')
	shutil.rmtree('datasetTweet/rassicuranti')
	os.mkdir('datasetTweet/neutri')
	os.mkdir('datasetTweet/allarmisti')
	os.mkdir('datasetTweet/rassicuranti')
	
	contatore = 0
	for x in result:
		if x[1] == 'ALLARM':
			f=open('datasetTweet\\allarmisti\\'+str(contatore), "w")
			f.write(x[0])
			f.close()
		elif x[1] == 'RASSIC':
			f=open('datasetTweet\\rassicuranti\\'+str(contatore), "w")
			f.write(x[0])
			f.close()
		elif x[1] == 'NEUTRO':
			f=open('datasetTweet\\neutri\\'+str(contatore), "w")
			f.write(x[0])
			f.close()
		contatore=contatore+1





	