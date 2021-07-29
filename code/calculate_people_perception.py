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
import GetOldTweets3 as got

# ----- PARAMETRI DI CONFIGURAZIONE -----
# start_date e end_date specificano l'intervallo temporale dove verranno caricati i tweets
start_date ='2020-01-30' #inclusa nell'intervallo temporale
end_date='2020-02-23' #esclusa dall'intervallo temporale

# ---------------------------------------

num_captured = 0
# cattura tweets
def streamTweets(tweets):
	global num_captured
	for tweet in tweets:
		sql = "INSERT IGNORE INTO provisory_table(id, date, username, hashtags, retweets,text) VALUES (%s, %s, %s, %s, %s, %s)"
		val = (tweet.id, tweet.date, tweet.username, tweet.hashtags, tweet.retweets, tweet.text)
		mycursor.execute(sql, val)
		mydb.commit()
		num_captured=num_captured+1
		print('caprured tweets = '+str(num_captured))
	
mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="tweet_coronavirus"
)
mycursor = mydb.cursor()
#le seguenti due righe sono commentate per evitare il download dei tweet, in quanto già presenti nel db
#tweetCriteria = got.manager.TweetCriteria().setQuerySearch('coronavirus OR #coronavirus OR #WuhanCoronavirus OR #CoronavirusOutbreak OR #coronaviruschina OR #coronaviruswuhan OR #ChinaCoronaVirus OR #nCoV OR #coronaviruses OR ChinaWuHan OR #nCoV2020 OR #nCov2019').setLang('it').setSince(start_date)
#got.manager.TweetManager.getTweets(tweetCriteria,streamTweets)

print('--- end of tweets capture phase ---')

stopw_list = myUtility.loadStopWlist()

#carica blacklist
blacklist=list()
f=open('blacklist.txt', 'r')
line=f.readline().rstrip("\n")
while line: 
	blacklist.append(line)
	line=f.readline().rstrip("\n")
f.close()

#carica blacklist_regexp
blacklist_regexp=list()
f=open('blacklist_regexp.txt', 'r')
line=f.readline().rstrip("\n")
while line: 
	blacklist_regexp.append(line)
	line=f.readline().rstrip("\n")
f.close()

#carica word_blacklist
word_blacklist=list()
f=open('blacklist_words.txt', 'r')
line=f.readline().rstrip("\n")
while line: 
	word_blacklist.append(line)
	line=f.readline().rstrip("\n")
f.close()

#lista di date
date_list =list()
sql = "select distinct DATE(date) from provisory_table"
mycursor.execute(sql)
row = mycursor.fetchone()
while row is not None:
	date_list.append(row[0])
	row = mycursor.fetchone()

da_scrivere= "DATE;TOTAL_OF_TWEETS;NUM_RASSOC;NUM_ALLARM;NUM_NEUTRAL"+"\n"
f=open('classification_results.txt', "a")
f.write(da_scrivere)
f.close()

date = '2020-01-30'
#decommnetare la seguente riga e aggiustare l'indentazione per fare l'analisi sull'intero periodo specificato in start_date e end_date
#for date in date_list:
sql = "select text, id, username from provisory_table where DATE(date)='"+str(date)+"'"
mycursor.execute(sql)
tweetList = list()
idList = list()
username_to_delete_list=list() 

result=mycursor.fetchall()
z=0
for row in result:
	print("inizio giro" +str(z))
	username = row[2]
	stop=0
	
	#controllo blacklist_regexp
	for w in blacklist_regexp:
		lowc=username.lower()
		if re.search(w, lowc): 
			stop=1
			break
	#controllo blacklist
	for w in blacklist:
		if re.match(username, w): 
			stop=1
			break
	
	#controllo word_blacklist
	tweet_text = row[0].lower()
	s = re.split("\s+|'+", tweet_text)
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
		b=b+a+" "
	lowc=b
	if lowc.isspace(): 
		row = mycursor.fetchone()
		continue
	for w in word_blacklist:
		lo = re.split("\s+|'+", lowc)
		for l in lo:
			if re.match(" "+l+" ", " "+w+" "):
				stop=2
				break
	
	if stop == 1:
		if row[2] not in username_to_delete_list:
			username_to_delete_list.append(row[2])
	if stop == 0:
		tweetList.append(lowc)
		idList.append(row[1])
	
	print("fine giro "+str(z))
	z=z+1

print(len(tweetList))
#	for row in username_to_delete_list:
#		sql="delete from provisory_table where username='"+str(row)+"'"
#		mycursor.execute(sql)
#		mydb.commit()
	
clf = load('modello/modello_classificatore.joblib') 
count_vect = load('modello/count_vect.joblib') 
tfidf_transformer = load('modello/tfidf_transformer.joblib') 
k_feat = load('modello/k_feat.joblib') 



testingData_noLinkENum = myUtility.tweetPreprocessing(tweetList, stopw_list)
testingData_stemmed = myUtility.semming(testingData_noLinkENum)
print(len(testingData_stemmed))
X_new_counts = count_vect.transform(testingData_stemmed)#tokenization and word counting
X_new_tfidf = tfidf_transformer.transform(X_new_counts)#feature extraction
X_new_kf = k_feat.transform(X_new_tfidf)

predicted = clf.predict(X_new_kf)#prediction

mum_allarmisti =0  #classe 0
num_neutri =0      #classe 1
num_rassicuranti=0 #classe 2

i=0
for x in predicted:
	if x==0:
		mum_allarmisti=mum_allarmisti+1
		#sql="update tweet_coronavirus.etichettati_dal_classificatore set etichetta='ALLARM' where id="+str(idList[i])
		#mycursor.execute(sql)
		mydb.commit()
	elif x==1:
		num_neutri=num_neutri+1
		#sql="update tweet_coronavirus.etichettati_dal_classificatore set etichetta='NEUTRO' where id="+str(idList[i])
		#mycursor.execute(sql)
		mydb.commit()
	elif x==2:
		num_rassicuranti=num_rassicuranti+1
		#sql="update tweet_coronavirus.etichettati_dal_classificatore set etichetta='RASSIC' where id="+str(idList[i])
		#mycursor.execute(sql)
		mydb.commit()
	i=i+1
	
'''
sql="insert ignore into tweet_coronavirus.labeled_tweets(id,date,username,hashtags,retweets,text,etichetta) select * FROM tweet_coronavirus.etichettati_dal_classificatore where DATE(date)='"+str(date)+"'"
mycursor.execute(sql)
mydb.commit()
'''


print("TOTAL OF TWEETS = "+str(i))
print("%ALARMISTS   = "+str(mum_allarmisti/i))
print("%NEUTRAL     = "+str(num_neutri/i))
print("%REASSURING  = "+str(num_rassicuranti/i))

da_scrivere= str(date)+";"+str(i)+";"+str(num_rassicuranti)+";"+str(mum_allarmisti)+";"+str(num_neutri)+"\n"

f=open('classification_results.txt', "a")
f.write(da_scrivere)
f.close()