#import flask
from flask import Flask,request,abort
import json
from app.Config import *
import requests

#Line Bot API
from linebot import LineBotApi

#แปลงไฟล์ m4a เป็น wav
from pydub import AudioSegment
import os
from pathlib import Path

#แปลง เสียงเป็นข้อความ
import speech_recognition as sr

#โมเดล
import pandas as pd
import numpy as np

# from tqdm import tqdm
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from pythainlp.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from datetime import datetime

#google sheet
from oauth2client.service_account import ServiceAccountCredentials
import gspread
from random import randint


word2vec_model = KeyedVectors.load_word2vec_format('LTW2V_v0.1.bin',binary=True,unicode_errors='ignore')
df=pd.read_csv('data-project2.csv',names=['input_text','labels'])



app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():

    if request.method=='POST':
        print(request.json)
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        print(Reply_token)
        if payload['events'][0]['message']['type'] == "text" or payload['events'][0]['message']['type'] == "audio":
            if payload['events'][0]['message']['type'] == "text":
                message=payload['events'][0]['message']['text']
            elif payload['events'][0]['message']['type'] == "audio":
                # Contect with Line for audio
                message_id=payload['events'][0]['message']['id']
                file_path=os.path.abspath("./file_keep.m4a")
                line_bot_api=LineBotApi(Channel_access_token)
                message_content=line_bot_api.get_message_content(message_id)
                with open(file_path,'wb') as fd:
                    for chunk in message_content.iter_content():
                        fd.write(chunk)
                # Convert files m4a to wav
                m4a_filename=file_path
                wav_filename=os.path.abspath('./file_keep_wav.wav')

                # For Window in running program
                # AudioSegment.converter=os.path.abspath("./<Folder Name>/bin/ffmpeg.exe") #เปลี่ยน <Folder Name> เป็น Directory ของไฟล์ ffmpeg
                path=Path(m4a_filename)
                if path.is_file():
                    if m4a_filename is not None:
                        print(path,flush=True)
                        track=AudioSegment.from_file(m4a_filename,format="m4a")
                else:
                    print("file None")
                file_handle=track.export(wav_filename,format='wav')
                r=sr.Recognizer()
                audio_file=sr.AudioFile(wav_filename)
                with audio_file as source:
                    audio_data=r.record(source)
                try:
                    text_audio=r.recognize_google(audio_data,language='th')
                    message=text_audio
                    print("Transcription: " + text_audio)   # แสดงข้อความจากเสียงด้วย Google Speech Recognition
                except sr.RequestError as e:       # ประมวลผลแล้วไม่รู้จักหรือเข้าใจเสียง
                    message="not understand audio"
                    print("Could not understand audio")
            print(message)
            text=message

            if 'การเตรียมตัวก่อนและหลังผ่าตัดเต้านม' in message:
                dt = datetime.now()
                print("Deeplearning start: ",dt)
                Reply_message=Answer_sheet("การเตรียมตัวก่อนผ่าตัด")['answer']
                dt = datetime.now()
                print("Deeplearning end: ",dt)
            elif 'การรักษามะเร็งเต้านม' in message:
                Reply_message=Answer_sheet("วิธีการรักษาด้วยการผ่าตัด")['answer']
            elif 'อาการเบื้องต้นของมะเร็งเต้านม' in message:
                Reply_message=Answer_sheet("อาการเจ็บเต้านมเป็นมะเร็งเต้านมหรือเปล่า")['answer']
            elif 'ควรจะเริ่มทำแมมโมแกรมเมื่อไหร่' in message:
                Reply_message=Answer_sheet("อายุในการทำ Mammogram")['answer']
            elif 'การตรวจเต้านมด้วยตนเอง' in message:
                Reply_message=Answer_sheet("คลิปการตรวจเต้านมด้วยตนเอง")['answer']
            elif "not understand audio" in message:
                predict="ไม่เข้าใจคำถาม"
                TextMessage_text_Question=predict
                tmp=Answer_sheet(predict)
                TextMessage_text_Answer=tmp['answer']
                Reply_message_audio=tmp['url']
                Reply_message_audio_time=tmp['time']
            else:
                dt = datetime.now()
                print("Deeplearning start: ",dt)
                data_df=pd.DataFrame.from_records(df)

                #เปลี่ยนตัวอักษรตัวใหญ่เป็นตัวเล็ก
                data_df['cleaned_labels']=data_df['labels'].str.lower()

                #เอา labels ออก
                data_df.drop('labels',axis=1,inplace=True)

                data_df=data_df[data_df['cleaned_labels']!='garbage']


                cleaned_input_text=data_df['input_text'].str.strip()
                cleaned_input_text=cleaned_input_text.str.lower()

                data_df['cleaned_input_text']=cleaned_input_text
                data_df.drop('input_text',axis=1,inplace=True)

                data_df=data_df.drop_duplicates("cleaned_input_text",keep="first")

                input_text=data_df["cleaned_input_text"].tolist()
                labels=data_df["cleaned_labels"].tolist()
                train_text,test_text,train_labels,test_labels=train_test_split(input_text,labels,train_size=0.8,random_state=42)
                loaded_model = tf.keras.models.load_model('my_model') #ชื่อไฟล์ model ที่สร้างไว้
                text=text.lower()
                text=text.strip()
                # tokenize
                word_seq = word_tokenize(text)
                # map index
                word_indices = map_word_index(word_seq)
                # padded to max_leng
                padded_wordindices = pad_sequences([word_indices], maxlen=25, value=0)
                # predict to get logit
                logit = loaded_model.predict(padded_wordindices, batch_size=32)
                unique_labels = set(train_labels)
                index_to_label = [label for label in sorted(unique_labels)]
                #Check probability that low(0.75)
                index=[ logit[0][pred] for pred in np.argmax(logit, axis=1) ][0]
                if index<=0.75:
                    print("No")
                    predict="ไม่เข้าใจคำถาม"
                    if payload['events'][0]['message']['type'] == "audio":
                        #Get data with Google Sheet
                        tmp=Answer_sheet(predict)
                        TextMessage_text_Question='คำถามของคุณ: '+message
                        TextMessage_text_Answer=tmp['answer']
                        Reply_message_audio=tmp['url']
                        Reply_message_audio_time=tmp['time']
                    elif payload['events'][0]['message']['type'] == "text":
                        tmp=Answer_sheet(predict)
                        Reply_message=tmp['answer']
                else:
                    # get prediction
                    predict = [ index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]
                    print(predict)
                    if payload['events'][0]['message']['type'] == "audio":
                        tmp=Answer_sheet(predict)
                        TextMessage_text_Question='คำถามของคุณ: '+message
                        TextMessage_text_Answer=tmp['answer']
                        Reply_message_audio=tmp['url']
                        Reply_message_audio_time=tmp['time']
                        print("url",Reply_message_audio)
                        print("time",Reply_message_audio_time)
                    elif payload['events'][0]['message']['type'] == "text":
                        tmp=Answer_sheet(predict)
                        Reply_message=tmp['answer']
                dt = datetime.now()
                print("Deeplearning end: ",dt)
            
            
        else:
            predict="ไม่เข้าใจคำถาม"
            tmp=Answer_sheet(predict)
            Reply_message=tmp['answer']
        if payload['events'][0]['message']['type'] == "audio":
            ReplyMessage_audio(Reply_token,TextMessage_text_Question,TextMessage_text_Answer,Reply_message_audio,Reply_message_audio_time,Channel_access_token)
            Remove_File(m4a_filename)
            file_handle.close()
            Remove_File(wav_filename)
        elif payload['events'][0]['message']['type'] == "text":

            ReplyMessage(Reply_token,Reply_message,Channel_access_token)
        else:
            ReplyMessage(Reply_token,Reply_message,Channel_access_token)

        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!",200
    else:
        abort(400)
    
def Remove_File(path):
    if os.path.isfile(path):
        os.remove(path)
        print("File has been deleted")
    else:
        print("File does not exist")

def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200

def ReplyMessage_audio(Reply_token,TextMessage_text_Question,TextMessage_text_Answer,TextMessage_audio,Reply_message_audio_time,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }
    if TextMessage_text_Question =="ไม่เข้าใจคำถาม":
        data={
        "replyToken":Reply_token,
        "messages":[
        {
            "type":"text",
            "text":TextMessage_text_Answer
        },
        {
            "type":"audio",
            "originalContentUrl":TextMessage_audio,
            "duration": Reply_message_audio_time
        }
        ]
        }
    else:
        data={
            "replyToken":Reply_token,
            "messages":[{
                "type":"text",
                "text":TextMessage_text_Question
            },
            {
                "type":"text",
                "text":TextMessage_text_Answer
            },
            {
                "type":"audio",
                "originalContentUrl":TextMessage_audio,
                "duration": Reply_message_audio_time
            }
            ]
        }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200

def map_word_index(word_seq):
 
  indices = []
  for word in word_seq:
    if word in word2vec_model.vocab:
      indices.append(word2vec_model.vocab[word].index + 1)
    else:
      indices.append(1)
  return indices

# Get answer for google sheet
def Answer_sheet(predict):
    scope = ['https://www.googleapis.com/auth/spreadsheets']
                                                                   
    credentials = ServiceAccountCredentials.from_json_keyfile_name('<Key Flie>', scope) #ใส่ <Key Flie> เป็นชื่อไฟล์ Key ที่เป็น JSON จากการสร้างใน Google Cloud
    client = gspread.authorize(credentials)
    sheet=client.open_by_url("<Link>") #ใส่ <Link> เป็น Link ของ Google Sheet ที่แชร์ไว้
    worksheet = sheet.get_worksheet(0)
    results = worksheet.get_all_records()
    tmp=[]
    for i in results:
        if predict.lower() in i['class'].lower():
            tmp.append(i)
    random=randint(0,len(tmp)-1)
    time=tmp[random]['time']
    tmp_time=time.split(":")
    millisec_min=int(tmp_time[0])*60000
    millisec_sec=int(tmp_time[1])*1000
    millisec=millisec_min+millisec_sec
    url=tmp[random]['url_audio']
    answer=tmp[random]['answer']

    return {"time":millisec,"url":url,"answer":answer}

