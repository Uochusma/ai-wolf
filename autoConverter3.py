# -*- coding: utf-8 -*-
import os
import codecs
import sys
import io
import h5py
print(sys.stdout.encoding) # ANSI_X3.4-1968 等を出力
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
print(sys.stdout.encoding) # ANSI_X3.4-1968 等を出力
cwd = os.getcwd()
wordcnt = 0
wordList = []
vocab_fileName = 'vocab.txt'
def readVocab():
    f = codecs.open(vocab_fileName, 'r', 'utf-8')
    for i,word in enumerate(f.readlines()):
        wordList.append(word)
    f.close()
    return
readVocab()
wordcnt = len(wordList)
print('vocab=',len(wordList))
#=======
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical
import time
import MeCab
import numpy as np
import re
#========
t = Tokenizer()
t.fit_on_texts(wordList)# 全体の語彙を使用してTokenizerを実行
#=======
def checkText(aText):
    agentRegex = r'Agent\[(\d)+\]'
    agentRegex2 = r'>>AGENT\s*'
    r2 = r'[!"#$%&()*+,-./:;<=>?@^_`{|}~]\%'
    r3 = r'[【】「」“”—＝．～：；『』＋＜＝≠＞×■▲△○●↑←…]\％'
    text = re.sub(agentRegex,'AGENT',aText)
    text = re.sub(agentRegex2,'',text)
    text = re.sub(r2,'',text)
    text = re.sub(r3,'',text)
    text = text.replace('%','')
    text = text.replace('％','')
    text = text.replace('?','？')
    return(text)
maxUttrLen = 100#発言の最大単語数
train_dir1 = 'train'
chat_train_txt = 'chat_train.txt'
chat_train_prot = 'chat_prot.txt'
chat_trainList = []
def tokenizeUttr(aDirectory,aFile):
    uttrList = []
    m = MeCab.Tagger("-Owakati")
    f_comb = codecs.open(cwd+'/'+aDirectory+'/'+aFile,'r','utf-8')
    lines = f_comb.readlines()
    #agentRegex = r'Agent\[(\d)+\]'
    #agentRegex2 = r'>>AGENT\s*'
    for line in lines:
        #line = re.sub(agentRegex,'AGENT',line)
        #line = re.sub(agentRegex2,'',line)
        line = checkText(line)
        mecab_string = m.parse(line)
        mecab_list = mecab_string.split()
        traintts = t.texts_to_sequences(mecab_list)  # テキストの順番
        trainttm = t.texts_to_matrix(mecab_string)  # 表記
        maxlen = len(traintts)  # 議論スレッドの長さ（単語延べ数）
        #wordcnt = len(trainttm[0])  # 語彙サイズ（単語の異なり数）
        #print(line)
        #print(traintts)
        traintoc = to_categorical(traintts, wordcnt)# Onehot配列作成
        #print(np.shape(traintoc))
        trainnp = np.zeros((maxUttrLen - maxlen, wordcnt), 'int8')  # 議論スレッドの長さに満たない分の配列を作成する
        uttrList.append(np.append(traintoc.astype(np.int8), trainnp, axis=0))
        if(len(mecab_list)>maxUttrLen):
            print("max over!"*100)
    print(len(lines))
    return(uttrList)
chat_trainList = tokenizeUttr(train_dir1,chat_train_txt)
chat_uttrArray = np.reshape(chat_trainList,(len(chat_trainList),maxUttrLen,wordcnt))
#
actionS = ['estimate','comingout','divined','vote','None']
agentS = ['AGENT','None']
roleS = ['villager','seer','wolf','possesed','None']
specieS = ['human','wolf','None']
ACTION_NUM = len(actionS)
AGENT_NUM = len(agentS)
ROLE_NUM = len(roleS)
SPECIES_NUM = len(specieS)
def tokenizeProtocol(aDirectory,aFile):
    f_comb = codecs.open(cwd+'/'+aDirectory+'/'+aFile,'r','utf-8')
    lines = f_comb.readlines()
    #agentRegex = r'Agent\[(\d)+\]'
    #agentRegex2 = r'>>AGENT\s*'
    actionList = []
    subjectList = []
    targetList = []
    roleList = []
    speciesList = []
    for line in lines:
        protocol = line.split()
        if(len(protocol)==1):
            protocol = protocol[0].split(',')
        #
        actionArray = np.zeros(ACTION_NUM)
        actionArray[actionS.index(protocol[0])] = 1
        actionList.append(actionArray)
        #
        subjectArray = np.zeros(AGENT_NUM)
        #subject = re.sub(agentRegex,'AGENT',protocol[1])
        #subject = re.sub(agentRegex2,'',subject)
        subject = checkText(protocol[1])
        subjectArray[agentS.index(subject)] = 1
        subjectList.append(subjectArray)
        #
        targetArray = np.zeros(AGENT_NUM)
        #target = re.sub(agentRegex,'AGENT',protocol[2])
        #target = re.sub(agentRegex2,'',target)
        target = checkText(protocol[2])
        targetArray[agentS.index(target)] = 1
        targetList.append(targetArray)
        #
        roleArray = np.zeros(ROLE_NUM)
        roleArray[roleS.index(protocol[3])] = 1
        roleList.append(roleArray)
        #
        speciesArray = np.zeros(SPECIES_NUM)
        speciesArray[specieS.index(protocol[4])]
        speciesList.append(speciesArray)
        #
    actionArray2 = np.reshape(actionList,(len(lines),ACTION_NUM))
    subjectArray2 = np.reshape(subjectList,(len(lines),AGENT_NUM))
    targetArray2 = np.reshape(targetList,(len(lines),AGENT_NUM))
    roleArray2 = np.reshape(roleList,(len(lines),ROLE_NUM))
    speciesArray2 = np.reshape(speciesList,(len(lines),SPECIES_NUM))
    return([actionArray2,subjectArray2,targetArray2,roleArray2,speciesArray2])
chat_protArray = tokenizeProtocol(train_dir1,chat_train_prot)
#========
train_dir2 = 'train3'
divine_train_txt = 'divine_train2.txt'
divine_trainList = tokenizeUttr(train_dir2,divine_train_txt)
divine_uttrArray = np.reshape(divine_trainList,(len(divine_trainList),maxUttrLen,wordcnt))
divine_train_prot = 'divine_prot2.txt'
divine_protArray = tokenizeProtocol(train_dir2,divine_train_prot)
#
co_train_txt = 'co_train.txt'
co_trainList = tokenizeUttr(train_dir2,co_train_txt)
co_uttrArray = np.reshape(co_trainList,(len(co_trainList),maxUttrLen,wordcnt))
co_train_prot = 'co_prot.txt'
co_protArray = tokenizeProtocol(train_dir2,co_train_prot)
#
vote_train_txt = 'vote_train.txt'
vote_trainList = tokenizeUttr(train_dir2,vote_train_txt)
vote_uttrArray = np.reshape(vote_trainList,(len(vote_trainList),maxUttrLen,wordcnt))
vote_train_prot = 'vote_prot.txt'
vote_protArray = tokenizeProtocol(train_dir2,vote_train_prot)
#
comb1_train_txt = 'comb_Uttr_Train.txt'
comb1_trainList = tokenizeUttr(train_dir2,comb1_train_txt)
comb1_uttrArray = np.reshape(comb1_trainList,(len(comb1_trainList),maxUttrLen,wordcnt))
comb1_train_prot = 'comb_Prot_Train.txt'
comb1_protArray = tokenizeProtocol(train_dir2,comb1_train_prot)
#===========
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Reshape, Embedding, Flatten, Dropout, RepeatVector
from keras.layers import Concatenate
from keras.layers import LSTM, Bidirectional
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from keras.models import load_model
class P2Tmodel:
    def __init__(self):
        actionInputs = Input(shape=(ACTION_NUM,))
        subjectInputs = Input(shape=(AGENT_NUM,))
        targetInputs = Input(shape=(AGENT_NUM,))
        roleInputs = Input(shape=(ROLE_NUM,))
        speciesInputs = Input(shape=(SPECIES_NUM,))
        protocolInputs = Concatenate()([actionInputs,subjectInputs,targetInputs,roleInputs,speciesInputs])
        #lstm1 = LSTM(100)(protocolInputs)
        dense1 = Dense(100)(protocolInputs)
        repeatInputs = RepeatVector(maxUttrLen)(dense1)
        lstm2 = LSTM(50,return_sequences=True)(repeatInputs)
        remarkOut = Dense(wordcnt)(lstm2)
        self.model = Model(inputs=[actionInputs, subjectInputs,targetInputs,roleInputs,speciesInputs], outputs=[remarkOut])
        return
    #
    def compile(self):
        self.model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return
    #
    def train(self,aInputList,aOutput):
        batch_size = 5
        epochs=5
        self.model.fit(aInputList,aOutput,batch_size=batch_size, epochs=epochs, verbose=1)
        return
    #
    def save(self):
        self.model.save('modelP2T.h5')
        self.model.save_weights('weightP2T.h5')
        return
    #
    def load(self):
        self.model = load_model('modelP2T.h5')
        self.model.load_weights('weightP2T.h5', by_name=True)
        return
    #
    def tokenizeProtBatch(self,aProtocol):
        #agentRegex = r'Agent\[(\d)+\]'
        #agentRegex2 = r'>>AGENT\s*'
        #
        actionList = []
        actionArray = np.zeros(ACTION_NUM)
        actionArray[actionS.index(aProtocol[0])] = 1
        actionList.append(actionArray)
        actionArray2 = np.reshape(actionList,(1,ACTION_NUM))
        #
        subjectList = []
        subjectArray = np.zeros(AGENT_NUM)
        #subject = re.sub(agentRegex,'AGENT',aProtocol[1])
        #subject = re.sub(agentRegex2,'',subject)
        subject = checkText(aProtocol[1])
        subjectArray[agentS.index(subject)] = 1
        subjectList.append(subjectArray)
        subjectArray2 = np.reshape(subjectList,(1,AGENT_NUM))
        #
        targetList = []
        targetArray = np.zeros(AGENT_NUM)
        #target = re.sub(agentRegex,'AGENT',aProtocol[2])
        #target = re.sub(agentRegex2,'',target)
        target = checkText(aProtocol[2])
        targetArray[agentS.index(target)] = 1
        targetList.append(targetArray)
        targetArray2 = np.reshape(targetList,(1,AGENT_NUM))
        #
        roleList = []
        roleArray = np.zeros(ROLE_NUM)
        roleArray[roleS.index(aProtocol[3])] = 1
        roleList.append(roleArray)
        roleArray2 = np.reshape(roleList,(1,ROLE_NUM))
        #
        speciesList = []
        speciesArray = np.zeros(SPECIES_NUM)
        speciesArray[specieS.index(aProtocol[4])]
        speciesList.append(speciesArray)
        speciesArray2 = np.reshape(speciesList,(1,SPECIES_NUM))
        #
        return([actionArray2,subjectArray2,targetArray2,roleArray2,speciesArray2])

    def sample(self,aPreds,aTemperature=1.0):
        #print(aPreds)
        for i,val in enumerate(aPreds):
            if(val<0):
                aPreds[i] = 0
            elif(val>1.0):
                aPreds[i] = 1.0
        aPreds = np.asarray(aPreds).astype('float64')#aPreds は0.0~1.0の値を取る実数配列
        exp_preds = np.exp(aPreds)
        aPreds = exp_preds/np.sum(exp_preds)#aPredsの値の合計が１になるように正規化を行う
        aPreds = aPreds/np.sum(aPreds)
        probas = np.random.multinomial(1,aPreds,1)#multinomial 関数によって aPreds 内の値に応じた確率で試行を行って one-hot ベクトルにするs
        return(np.argmax(probas))#numpy.argmax() で最大値要素の内で最も小さいインデックスを返す つまり、１である要素の添字を返す
    #
    def mat2Index(self,aMatrix):
        indexS = []
        #print(np.shape(aMatrix))
        #print(aMatrix)
        for vec in aMatrix[0]:
            indexS.append(self.sample(vec))
        return(indexS)

    #
    def token2Text(self,aToken):
        uttr = ''
        indexS = self.mat2Index(aToken)
        for index in indexS:
            uttr+=wordList[index]
        return(uttr)
    #
    def predict(self,aProtocol):
        prot = self.tokenizeProtBatch(aProtocol)
        pred = self.model.predict(prot)
        text = self.token2Text(pred)
        return(text)
#===========================================
p2tModel = P2Tmodel()
p2tModel.compile()
print(np.shape(chat_uttrArray))
print(np.shape(chat_protArray[0]))
#p2tModel.train(chat_protArray,chat_uttrArray)
print("Train divine_prot -> divine_uttr start")
p2tModel.train(divine_protArray,divine_uttrArray)
print("Train divine_prot->divine_uttr done")
print("Train co_prot -> co_uttr start")
p2tModel.train(co_protArray,co_uttrArray)
print("Train co_prot -> co_uttr done")
print("Train vote_prot -> vote_uttr start")
p2tModel.train(vote_protArray,vote_uttrArray)
print("Train vote_prot -> vote_uttr done")
print(p2tModel.predict(['comingout','None','None','seer','None']))
#===========================================
class T2Pmodel:
    def __init__(self,aTokenizer):
        self.tokenizer = aTokenizer
        remarkInput = Input(shape=(maxUttrLen,wordcnt))
        lstm1 = LSTM(100)(remarkInput)
        actionOut = Dense(ACTION_NUM)(lstm1)
        subjectOut = Dense(AGENT_NUM)(lstm1)
        targetOut = Dense(AGENT_NUM)(lstm1)
        roleOut = Dense(ROLE_NUM)(lstm1)
        speciesOut = Dense(SPECIES_NUM)(lstm1)
        self.model = Model(inputs=[remarkInput], outputs=[actionOut, subjectOut,targetOut,roleOut,speciesOut])
        return
    #
    def compile(self):
        self.model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()
        return
    #
    def train(self,aInput,aOutputList):
        batch_size = 5
        epochs=5
        self.model.fit(aInput,aOutputList,batch_size=batch_size, epochs=epochs, verbose=1)
        return
    #
    def save(self):
        self.model.save('modelT2P.h5')
        self.model.save_weights('weightT2P.h5')
        return
    #
    def load(self):
        self.model = load_model('modelT2P.h5')
        self.model.load_weights('weightT2P,h5', by_name=True)
        return
    #
    def tokenizeUttrBatch(self,aText):
        m = MeCab.Tagger("-Owakati")
        #agentRegex = r'Agent\[(\d)+\]'
        #agentRegex2 = r'>>AGENT\s*'
        #text = re.sub(agentRegex,'AGENT',aText)
        #text = re.sub(agentRegex2,'',text)
        text = checkText(aText)
        mecab_string = m.parse(text)
        mecab_list = mecab_string.split()
        traintts = self.tokenizer.texts_to_sequences(mecab_list)  # テキストの順番
        trainttm = self.tokenizer.texts_to_matrix(mecab_string)  # 表記
        maxlen = len(traintts)  # 議論スレッドの長さ（単語延べ数）
        #print(mecab_list)
        #print(traintts)
        traintoc = to_categorical(traintts, wordcnt)# Onehot配列作成
        trainnp = np.zeros((maxUttrLen - maxlen, wordcnt), 'int8')  # 議論スレッドの長さに満たない分の配列を作成する
        uttrArray = np.append(traintoc.astype(np.int8), trainnp, axis=0)
        return(uttrArray)
    #
    def sample(self,aPreds,aTemperature=1.0):
        for i,val in enumerate(aPreds):
            if(val<0):
                aPreds[i] = 0
            elif(val>1.0):
                aPreds[i] = 1.0
        aPreds = np.asarray(aPreds).astype('float64')#aPreds は0.0~1.0の値を取る実数配列
        exp_preds = np.exp(aPreds)
        aPreds = exp_preds/np.sum(exp_preds)#aPredsの値の合計が１になるように正規化を行う
        aPreds = aPreds/np.sum(aPreds)
        probas = np.random.multinomial(1,aPreds,1)#multinomial 関数によって aPreds 内の値に応じた確率で試行を行って one-hot ベクトルにするs
        return(np.argmax(probas))#numpy.argmax() で最大値要素の内で最も小さいインデックスを返す つまり、１である要素の添字を返す
    #
    def token2Protocol(self,aTokenS):
        index1 = self.sample(aTokenS[0][0])
        action = actionS[index1]
        subject = agentS[self.sample(aTokenS[1][0])]
        target = agentS[self.sample(aTokenS[2][0])]
        role = roleS[self.sample(aTokenS[3][0])]
        species = specieS[self.sample(aTokenS[4][0])]
        return([action,subject,target,role,species])
    #
    def predict(self,aText):
        #"""
        protocol = []
        lines = aText.split("。")
        if(len(lines)>2):
            lines.pop(-1)
            for tex in lines:
                text = self.tokenizeUttrBatch(tex)
                textBatch = [text]
                textArray = np.reshape(textBatch,(1,maxUttrLen,wordcnt))
                pred = self.model.predict(textArray)
                protocol.append(self.token2Protocol(pred))
        else:
            text = self.tokenizeUttrBatch(aText)
            textBatch = [text]
            textArray = np.reshape(textBatch,(1,maxUttrLen,wordcnt))
            pred = self.model.predict(textArray)
            protocol.append(self.token2Protocol(pred))
        #""""
        """
        text = self.tokenizeUttrBatch(aText)
        textBatch = [text]
        textArray = np.reshape(textBatch,(1,maxUttrLen,wordcnt))
        pred = self.model.predict(textArray)
        protocol = self.token2Protocol(pred)
        """
        return(protocol)
#=============
t2pModel = T2Pmodel(t)
t2pModel.compile()
print("Train chat_uttr -> chat_prot start")
t2pModel.train(chat_uttrArray,chat_protArray)
print("Train chat_uttr -> chat_prot done")
t2pModel.train(divine_uttrArray,divine_protArray)
t2pModel.train(vote_uttrArray,vote_protArray)
t2pModel.train(co_uttrArray,co_protArray)
t2pModel.train(comb1_uttrArray,comb1_protArray)
#p =t2pModel.predict('COするけど、ぼく占い師なんだよね。占いの結果だけど、Agent[03]は白だったよ。')
#print(p)
#=========================
maxProt = 100
preDir = '/train5'
f_prot = codecs.open(cwd+preDir+'/comb_Prot_Train.txt', 'w', 'utf-8')
f_uttrR = codecs.open(cwd+'/comb_Uttr_Train.txt','r','utf-8')
f_uttrW = codecs.open(cwd+preDir+'/comb_Uttr_Train.txt', 'w', 'utf-8')
#f_prot = open(cwd+preDir+'/comb_Prot_Train.txt', 'w')
#f_uttrW = open(cwd+preDir+'/comb_Prot_Train.txt', 'w')
lines = f_uttrR.readlines()
#agentRegex = r'Agent\[(\d)+\]'
#agentRegex2 = r'>>AGENT\s*'
for i,line in enumerate(lines):
    #line = re.sub(agentRegex,'AGENT',line)
    #line = re.sub(agentRegex2,'',line)
    line = checkText(line)
    splitline = line.split("。")
    if(len(splitline)>2):
            splitline.pop(-1)
            for tex in splitline:
                print(tex)
                print(type(tex))
                f_uttrW.write(tex)
                f_uttrW.write('\n')
    else:
        tex = line.replace('\n','')
        #print(tex)
        #print(type(tex))
        f_uttrW.write(tex)
        f_uttrW.write('\n')
    pList =t2pModel.predict(line)
    print(pList)
    for p in pList:
        #print()
        f_prot.write(','.join(p))
        f_prot.write('\n')
    """"
    pList = []
    if(len(lines)>2):
        lines.pop(-1)
        for tex in lines:
            p =t2pModel.predict(tex)
            pList.append(p)
    else:
        p =t2pModel.predict(line)
        pList.append(p)
    #
    for p in pList:
        print(p)
        f_prot.write(','.join(p))
        f_prot.write('\n')
    if(i+1>maxProt):
        break
    """
f_prot.close()
f_uttrR.close()
f_uttrW.close()
#==========================================
train_dir4 = 'train4'
comb4_train_txt = 'comb_Uttr_Train.txt'
comb4_trainList = tokenizeUttr(train_dir4,comb4_train_txt)
comb4_uttrArray = np.reshape(comb4_trainList,(len(comb4_trainList),maxUttrLen,wordcnt))
comb4_train_prot = 'comb_Prot_Train.txt'
comb4_protArray = tokenizeProtocol(train_dir4,comb4_train_prot)
#
t2pModel.train(comb4_uttrArray,comb4_protArray)
t2pModel.save()
#
p2tModel.train(comb4_protArray,comb4_uttrArray)
print(p2tModel.predict(['comingout','None','None','seer','None']))
p2tModel.save()