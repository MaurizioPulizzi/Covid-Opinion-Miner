#questo file crea una lista di termini presenti tra i tweets scartati ma non presenti nei tweets etichettati
#usa un file idEtichettati.txt che deve contenere tutti gli id dei tweet scanzionati


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
import myUtility

stopw_list = myUtility.loadStopWlist()

#carica blacklist
blacklist=list()
f=open('blacklist.txt', 'r')
line=f.readline().rstrip("\n")

while line: 
	blacklist.append(line)
	line=f.readline().rstrip("\n")
f.close()

#carica blacklist
letti=list()
f=open('idEtichettati.txt', 'r')
line=f.readline().rstrip("\n")

while line: 
	letti.append(line)
	line=f.readline().rstrip("\n")
f.close()

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="tweet_coronavirus"
)
wordlist_ok = list()
wordlist_no = list()
id_etich = list()
mycursor = mydb.cursor()
sql = "SELECT text, id FROM table_9"
mycursor.execute(sql)
result= mycursor.fetchall()
print(len(result))



for x in result:
	id_etich.append(x[1].rstrip("\n"))
	words = str(x[0]).lower()
	w = words.split()
	for a in w:
		if a in wordlist_ok:
			continue
		wordlist_ok.append(a)
	wordlist_ok = list(dict.fromkeys(wordlist_ok))

wordlist_ok_clean=list()	
for a in wordlist_ok:
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
	wordlist_ok_clean.append(a)
	
wordlist_ok_clean = list(dict.fromkeys(wordlist_ok_clean))
wordlist_ok_clean.sort()

id_etich.sort()
letti.sort()
letti = list(dict.fromkeys(letti))
id_etich = list(dict.fromkeys(id_etich))
for idl in letti:
	if idl in id_etich:
		continue
	sql = "SELECT text FROM tweet_prima_del_30_genn where id='"+str(idl)+"'"
	mycursor.execute(sql)
	result= mycursor.fetchone()
	words = str(result[0]).lower()
	w = words.split()
	for a in w:
		if a in wordlist_no or a in wordlist_ok_clean:
			continue
		wordlist_no.append(a+"\n")
	wordlist_no = list(dict.fromkeys(wordlist_no))

wordlist_no_clean=list()	
for a in wordlist_no:
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
	wordlist_no_clean.append(a)

	
wordlist_no_clean.sort()
wordlist_ok_clean.sort()	
wordlist_no_clean = list(dict.fromkeys(wordlist_no_clean))
wordlist_ok_clean = list(dict.fromkeys(wordlist_ok_clean))	

stemmer = SnowballStemmer("italian")
trainingData_stemmed=[]
stemmato=[]
for testo in wordlist_no_clean:
	stemmato.append(stemmer.stem(testo))
	
stemmato = list(dict.fromkeys(stemmato))
stemmato.sort()	

f=open('black_w_list.txt', "a")


for a in wordlist_no_clean:
	if a in wordlist_ok_clean:
		continue
	f.write(a+"\n")


f.close()