 #passiamo i dati etichettati da mysql ai file pronti da usare per addestrare la rete

import mysql.connector

mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="tweetcoronavirus"
)
mycursor = mydb.cursor()
sql = "SELECT text, etichetta FROM tweetcoronavirus.table_8"
mycursor.execute(sql)
result= mycursor.fetchall()

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