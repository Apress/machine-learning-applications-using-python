# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:20:18 2018

@author: PUNEETMATHUR
"""
#Loading tkinter libraries which will be used in the UI of chatbot
import tkinter
from tkinter import *
from tkinter.scrolledtext import *
from tkinter import ttk
import time
from PIL import ImageTk, Image
import tkinter 

#Loading random for random choices in our chat program
import random

#Splash Screen 
splash = tkinter.Tk()

splash.title("Welcome to Applications of Machine Learning in Healthcare, Retail & Finance")
splash.geometry("1000x100")
splash.configure(background='green')
w = Label(splash, text="DirectPyth Diabetes Diagnostics Chatbot by Puneet Mathur\nLoading...",font=("Helvetica", 26),fg="white",bg="green")
w.pack()

splash.update()
time.sleep(6)
splash.deiconify()
splash.destroy()

#Initializing tkinter library for UI window showup
window = tkinter.Tk()
s = tkinter.Scrollbar(window)
chatmsg = tkinter.Text(window)
chatmsg.focus_set()
s.pack(side=tkinter.RIGHT, fill=tkinter.Y)
chatmsg.pack(side=tkinter.TOP, fill=tkinter.Y)
s.config(command=chatmsg.yview)
chatmsg.config(yscrollcommand=s.set)
input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=tkinter.BOTTOM, fill=tkinter.X)
bot_text="Welcome to DirectPyth Diagnostic Center\n"
chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
bot_text = "Press enter to continue "
chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
chatmsg.focus()

#Diagnostics Corpus for the chatbot
greet=['Hello welcome to DirectPyth','Hi welcome to DirectPyth','Hey welcome to DirectPyth','Good day to you welcome to DirectPyth']
confirm=['yes','yay','yeah','yo']
memberid=['12345','12346','12347','12348','12349']
customer = ['hello','hi','hey']
answer = ['I uderstand you feel happy but please stay to the point and select one of the options',"I sympathize with you, However please do not deviate from the topic"]
greetings = ['hola Welcome to DirectPyth again', 'hello Welcome to DirectPyth again', 'hi Welcome to DirectPyth again', 'Hi Welcome to DirectPyth again', 'hey! Welcome to DirectPyth again', 'hey Welcome to DirectPyth again']
question = ['how are you?', 'how are you doing?']
responses = ['Okay', "I'm fine"]
tests=['Type 1 for hbA1c test', "Type 2 for Blood Viscosity test","Type 3 for Heart rate test","Type 4 for Blood Oxygen test","Type 5 for Blood Pressure"]
testresponse= ['1','2','3','4','5','6']

#Global variable to check first time greeting  
firstswitch=1
newid="12310"
memid=0

def chat(event):
    import time
    import random
    global memid    
    condition=""
    #Greet for first time
    global firstswitch
    print(firstswitch)

    if (firstswitch==1):
        bot_text = random.choice(greet)
        chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
        bot_text = "If you are an existing Member of DirectPyth please enter your membershipid: or enter no if you are not a member"
        chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
        firstswitch=2
    if (firstswitch!=1):
        input_get = input_field.get().lower()
        if any(srchstr in input_get for srchstr  in memberid):
            memid=input_get
            bot_text = "Thank you for being a loyal member of DirectPyth\n Please choose a test from following menu to continue\nType 1 for hbA1c test\nType 2 for Blood Viscosity test\nType 3 for Heart rate test\nType 4 for Blood Oxygen test\nType 5 for Blood pressure test\nType 6 to exit\n\n"
        elif (input_get=="no"):
            memid=newid
            bot_text = "Your new Memberid is: " + newid + " Please remember this for future reference.\n Please choose a test from following menu to continue\nType 1 for hbA1c test\nType 2 for Blood Viscosity test\nType 3 for Heart rate test\nType 4 for Blood Oxygen test\nType 5 for Blood pressure test\nType 6 to exit\n\n"
        elif any(srchstr in input_get for srchstr  in testresponse):
        
                bot_text = "Please place any of your finger on the Finger panel above to conduct the test"
                chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
                delaycounter=0
                for delaycounter in range(0,10):
                    bot_text = str(delaycounter)
                    time.sleep(1)
                    chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
                bot_text = "Please wait generating your report\n"
                chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
                time.sleep(2)
                if (input_get=="1"):
                    hba1c=random.randint(4,10)
                    bot_text = "MemberID: " + str(memid) + " Your hbA1c test result is: " + str(hba1c)
                    if(hba1c>=4 and hba1c<=5.6):
                        condition="You are don't have diabetes"
                    elif(hba1c>=5.7 and hba1c<=6.4):
                        condition="You are Prediabetic"
                    elif(hba1c>=6.5):
                        condition="You are Diabetic"
                    bot_text=bot_text +  " Your condition is: " + condition    
                    chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)
                elif (input_get=="2"):
                    viscosity=random.randint(20,60)
                    bot_text = "MemberID: " + str(memid) + " Your Blood Viscosity level test result is: " + str(viscosity)
                elif (input_get=="3"):
                    heartrate=random.randint(40,150)
                    bot_text = "MemberID: " + str(memid) + " Your Heart rate test result is: " + str(heartrate)
                elif (input_get=="4"):
                    oxygen=random.randint(90,100)
                    bot_text = "MemberID: " + str(memid) + " Your Blood Oxygen level test result is: " + str(oxygen)
                elif (input_get=="5"):
                    systolic=random.randint(80,200)
                    diastolic=random.randint(80,110)
                    bot_text = "MemberID: " + str(memid) + " Your Blood Pressure test result is: Systolic: " + str(systolic) + " Diastolic: " + str(diastolic)
                elif (input_get=="6"):
                    import sys
                    window.deiconify()
                    window.destroy()
                    sys.exit(0)
        else:
         from nltk.stem import WordNetLemmatizer
         import nltk
         if((not input_get) or (int(input_get)<=0)):    
                        print("did you just press Enter?") #print some info
         else:
             lemmatizer = WordNetLemmatizer()
             input_get = input_field.get().lower()
             lemvalue=lemmatizer.lemmatize(input_get)
             whatsentiment=getSentiment(lemvalue)
             if (whatsentiment=="pos"):
                 bot_text = answer[0]
                 #print("Positive Sentiment")
             elif (whatsentiment=="neg"):
                 bot_text = answer[1]
             #print("Negative Sentiment")
             chatmsg.insert(INSERT, '%s\n' % lemvalue)
             #bot_text = "I did not understand what you said !"
            

        
    
    chatmsg.insert(INSERT, 'Bot:%s\n' % bot_text)    
    #label = Label(window, text=input_get)
    input_user.set('')
    #label.pack()
    return "break"

#Sentiment Analyzer using NLP
def getSentiment(text):
    import nltk
    from nltk.tokenize import word_tokenize

    #nltk.download('punkt')
    # Step 1 – Training data building the Diabetes corpus
    train = [("Thanks for an excellent report", "pos"),
    ("Your service is very quick and fast", "pos"),
    ("I am pleased with your service", "pos"),
    ("I did not know i was diabetic until you gave me this report", "neg"),
    ("Service - Little slow, probably because too many people.", "neg"),
    ("The place is not easy to locate", "neg"),
    ("The place is very easy to locate", "pos"),
    ("Not satisfied will take a second opinion", "neg"),
    ("No human contact everything is so robotic here", "neg"),
    ("can i talk to a human not convinced with your report", "neg"),
    ("good results", "pos"),
    ("good service", "pos"),
    ("great service", "pos"),
    ("excellent service", "pos"),
    ("amazing technology", "pos"),
    ("fast service and satisfying report", "pos"),
    ("your report sucks", "neg"),
    ("this report will cost me a fortune", "neg"),
    ("I have diabetes", "neg"),
    ("this report will cost me a fortune", "neg"),
    ("this report means i have a dreadful disease", "neg"),
    ("will i need to take new medication", "neg"),
    ("i need to take my insulin injections regularly", "neg"),
    ("my lipids are getting worst need to talk to the doctor", "neg"),
    ("oh my god very bad results", "neg"),
    ("bad service", "neg"),
    ("very bad service", "neg"),
    ("poor service", "neg"),
    ("very bad service", "neg"),
    ("slow service", "neg"),
    ("very slow service", "neg"),
    ("diabetes got worst is this report accurate", "neg"),
    ("i dont believe this report", "neg"),
    ("i dont like this report", "neg"),
    ("i am in a diabetic hell", "neg"),
    ]
    # Step 2 Tokenize the words to dictionary
    dictionary = set(word.lower() for passage in train for word in word_tokenize(passage[0]))
    # Step 3 Locate the word in training data
    t = [({word: (word in word_tokenize(x[0])) for word in dictionary}, x[1]) for x in train]
    # Step 4 – the classifier is trained with sample data
    classifier = nltk.NaiveBayesClassifier.train(t)
    test_data = "oh my god what is this"
    test_data_features = {word.lower(): (word in word_tokenize(test_data.lower())) for word in dictionary}
    print (classifier.classify(test_data_features))
    return classifier.classify(test_data_features)
    
#Start the program chat and put in loop
input_field.bind("<Return>", chat)
tkinter.mainloop()

