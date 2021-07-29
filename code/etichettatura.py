import random
import mysql.connector
import re
max_c = 25209 #numero di tweets nella tabella del db

allarm=0
rassic=0
neutri=0

totLink = 0
conLinkNonSign = 0
hasLink=0
fine = 0

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


mydb = mysql.connector.connect(
	host="localhost",
	user="root",
	passwd="",
	database="tweet_coronavirus"
)
mycursor = mydb.cursor()

listanum = list() 
file = open("idEtichettati.txt", "r")  #nel file ci sono scritti gli id gia etichettati e andiamo aggiungendo quelli che via via etichettiamo
line=file.readline().rstrip("\n")
while line: 
	listanum.append(int(line))
	line=file.readline().rstrip("\n")
file.close()
	
while fine == 0:
	num = random.randint(1, max_c)
	if num in listanum:
		continue
	else:
		listanum.append(num)

	sql = "SELECT * FROM tweet_coronavirus.tweet_prima_del_30_genn WHERE contatore = "+str(num)
	mycursor.execute(sql)

	res = mycursor.fetchone()
	
	username=str(res[3])
	stop=0
	#controllo blacklist
	for w in blacklist:
		if re.match(username, w): 
			stop=1
			break
	if stop == 1:
		continue
	#controllo blacklist_regexp
	for w in blacklist_regexp:
		lowc=username.lower()
		if re.search(w, lowc): 
			stop=1
			break
	if stop == 1:
		continue
	
	#controllo link
	testo=res[6]
	s=testo.split()
	for t in s:
		if re.search("http", t): 
			hasLink=1
			break
	if hasLink == 1:
		totLink =totLink+1
		file=open("totale_con_link.txt", "w")
		file.write(str(totLink))
		file.close()
	
	print("USERNAME: "+username)
	print(res[6])
	print('\n\n')

	tipo = input("TIPO?   ")

	if tipo == '-':
		etichetta = 'ALLARM'
		allarm = allarm+1
	elif tipo == '+':
		etichetta = 'RASSIC'
		rassic = rassic+1
		
	elif tipo == '0':
		etichetta = 'NEUTRO'
		neutri = neutri+1
	elif tipo == 'q':
		fine = 1
		break
	elif tipo == 's':
		etichetta = 'SCARTO'

	f=open("idEtichettati.txt", "a")
	f.write(str(res[0])+"\n")
	f.close()
	
	if hasLink == 1 and tipo == 's':
		conLinkNonSign = conLinkNonSign+1
		file=open("totale_con_link_non_significativi.txt", "w")
		file.write(str(conLinkNonSign))
		file.close()
	
	if tipo =='+' or tipo =='-' or tipo =='0':
		sql = "INSERT INTO etichettati_manualmente(id, contatore, date, username, hashtags, retweets,text, etichetta, haslink) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
		val = (res[0], res[1], res[2], res[3], res[4], res[5], res[6], etichetta, hasLink)
		mycursor.execute(sql, val)
		mydb.commit()
	hasLink=0
	print('\n rassic = '+str(rassic)+', allarm = '+str(allarm)+', neutri = '+str(neutri)+', con link = '+str(totLink)+', link scartati = '+str(conLinkNonSign/(1+totLink)))
	print('\n\n\n\n')
	tipo='s'
	