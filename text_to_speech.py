# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:28:46 2024

@author: Admin
"""
import os
#import google.cloud.texttospeech as tts
from google.cloud import texttospeech_v1
from playsound import playsound

def speak(text1):

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'kubernet-test-374818-4217d3f1f723.json'
    client = texttospeech_v1.TextToSpeechClient()
    print(type(text1))
    
    text = f"<speak>{text1}</speak>"
    synthesis_input = texttospeech_v1.SynthesisInput(ssml =text)
    
    voice1 = texttospeech_v1.VoiceSelectionParams(
        language_code = 'en-in',
        ssml_gender = texttospeech_v1.SsmlVoiceGender.MALE
        )
    
    audio_config= texttospeech_v1.AudioConfig(
        audio_encoding = texttospeech_v1.AudioEncoding.MP3
        )

#text = '''<speak> Hi..... I am Vivek Parihar, I am working on chatbot</speak>'''

    #synthesis_input = texttospeech_v1.SynthesisInput(ssml =text)
    
    response1 = client.synthesize_speech(
        input = synthesis_input,
        voice= voice1,
        audio_config= audio_config)
    
    with open('audio.mp3', 'wb', ) as output:
        output.write(response1.audio_content)
     
    
    playsound(r'C:\Users\Admin\OneDrive - bizmetric.com\Documents\edtech\pdfGPT-main\audio.mp3')
    print('playing sound using  playsound')
