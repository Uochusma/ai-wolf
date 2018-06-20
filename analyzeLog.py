# -*- coding: utf-8 -*-
import os
import codecs
import sys
import io
print(sys.stdout.encoding) # ANSI_X3.4-1968 等を出力
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(sys.stdout.encoding) # ANSI_X3.4-1968 等を出力
#from autoConverter import P2Tmodel,T2Pmodel
cwd = os.getcwd()
def analyzeLog():
	train_dir = ("/train2/")
	save_dir = ("/train2-2/")
	files = os.listdir(cwd+train_dir)
	print(len(files))
	f_comb = codecs.open(cwd+'/comb_Uttr_Train.txt','w','utf-8')
	for file in files:
	    #print(file)
	    if(file.find('.txt')==-1):
	    	continue
	    filaPath = cwd+train_dir+file
	    f = codecs.open(filaPath,'r','utf-8')
	    trainfile = file+'_train'+'.txt'
	    f2 = codecs.open(cwd+save_dir+trainfile,'w','utf-8')
	    lines = f.readlines()
	    for line in lines:
	    	data = line.split(',')
	    	state = data[1]
	    	if(state=='talk'):
	    		text = data[5]
	    		if(text!='Over\n' and text!='Skip\n'):
	    			step = data[2]
	    			agent= data[4]
	    			agentCount = data[3]
	    			#protocol = T2Pmodel.predict(text)
	    			#protocolStr=','.join(protocol)
	    			#trainLine = text+','+protocolStr
	    			trainLine = text
	    			f2.write(trainLine)
	    			f_comb.write(trainLine)
	    f2.close()
	#
	f_comb.close()
	return
#analyzeLog()
#==============================================================
### スレッドデータから語彙（単語の異なり数）を作成し訓練データを作成する
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical
import time
import MeCab
import numpy as np
import re
t = Tokenizer()
wordList = []
uttrLen = []
uttrList = []
def tokenizeUttr():
	m = MeCab.Tagger("-Owakati")
	f_comb = codecs.open(cwd+'/comb_Uttr_Train.txt','r','utf-8')
	lines = f_comb.readlines()
	agentRegex = r'Agent\[(\d)+\]'
	for line in lines:
		line = re.sub(agentRegex,'AGENT',line)
		mecab_string = m.parse(line)
		mecab_list = mecab_string.split()
		uttrList.append(mecab_list)
		for word in mecab_list:
			if word not in wordList:
				wordList.append(word)
		uttrLen.append(len(mecab_list))
	t.fit_on_texts(wordList)# 全体の語彙を使用してTokenizerを実行
	return(max(uttrLen))
maxUttrLen = tokenizeUttr()
voc_size = len(wordList)
print('voc_size=',voc_size)
print(wordList)
vocab_fileName = 'vocab.txt'
def writeVocab():
    f = codecs.open(vocab_fileName, 'w', 'utf-8')
    for word in wordList:
        f.write(word)
        f.write('\n')
    f.close()
    return
writeVocab()
#
trainList = []
def makeTrain():
	for i,line in enumerate(uttrList):
		traintts = t.texts_to_sequences(line)  # テキストの順番
		trainttm = t.texts_to_matrix(line)  # 表記
		maxlen = len(traintts)  # 議論スレッドの長さ（単語延べ数）
		wordcnt = len(trainttm[0])  # 語彙サイズ（単語の異なり数）
		print(i, traintts, wordcnt, line)
		traintoc = to_categorical(traintts, wordcnt)  # Onehot配列作成
		trainnp = np.zeros((maxUttrLen - maxlen, wordcnt), 'int8')  # 議論スレッドの長さに満たない分の配列を作成する
		trainList.append(trainnp)
	return(i+1)
uttrNum = makeTrain()
