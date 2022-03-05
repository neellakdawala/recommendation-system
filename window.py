from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk



import numpy as np
import pandas as pd
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import math


def search_paper(query):
    #query="Clustering"

    df=pd.read_csv("paper_data.csv")

    features = ['Title','Abstract','Author']
    for feature in features:
        df[feature] = df[feature].fillna('')

    def combined_features(row):
        return row['Title']+" "+row['Abstract']+" "+row['Author']
    df["combined_features"] = df.apply(combined_features, axis =1)


    corpus=[]
    corpus=df['combined_features'].tolist()



    corpus.append(query)


    x1=[]
    for x in corpus:
        x1.append(re.sub("[^a-zA-Z\s]","",x.lower()))

    y=[]
    for x in x1:
        y.append(nltk.word_tokenize(x))


    z=[]
    for x in y:
        z.append([word for word in x if word not in nltk.corpus.stopwords.words("english")])


    from nltk.stem import PorterStemmer
    pst=PorterStemmer()


    z2=[]
    for x in z:
        z2.append([pst.stem(word) for word in x])


    z1=[]
    for x in z2:
        str=' '.join(x)
        z1.append(str)
    query_stem=z1[len(z1)-1]
    z1=z1[:len(z1)-1]


    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()  
    vectorizer.fit(z1)
    vocab1=vectorizer.vocabulary_
    l=list(vocab1.items())
    l.sort()
    vocab=dict(l)


    z1.append(query_stem)
    vector = vectorizer.transform(z1)
    matrix=vector.toarray()


    matrix1=matrix.astype(np.float)


    vocab2= {value:key for key,value in vocab.items()}


    df1=pd.DataFrame(matrix)
    df1=df1.rename(columns=vocab2)


    num=len(matrix1[0])
    for x in range(len(matrix1)):
        maxi=matrix[x].max()
        for y in range(num):
            matrix1[x][y]=matrix[x][y]/maxi

            
            
    df2=pd.DataFrame(matrix1)
    df2=df2.rename(columns=vocab2)


    df_value=[]
    for x in df2.columns:
        df_value.append((df2[x]!=0).sum())




    N=len(corpus)
    idf_value=[]
    for x in df_value:
        idf_value.append(round(math.log(N/x,10),4))


    matrix3=matrix1
    for i in range(len(matrix1)):
        for j in range(num):
            matrix3[i][j]=matrix3[i][j]*idf_value[j]

            
            

    df3=pd.DataFrame(matrix3)
    df3=df3.rename(columns=vocab2)


    cosine_matrix=cosine_similarity(df3)
    df4=pd.DataFrame({'index':[x for x in range(len(df3))],'Cosine_Similarity':cosine_matrix[df.shape[0]]})


    df5=pd.merge(df,df4,how='left',on='index')

    df6=df5.sort_values('Cosine_Similarity',ascending=False)

    a=0
    for x in df6.index:
        print("RESEARCH PAPER:",x+1)
        print('Title:',df6['Title'][x])
        print('Abstract:',df6['Abstract'][x])
        print('Author:',df6['Author'][x])
        print('Citation:',df6['Citation'][x])
        print('\n')
        if a==4:
            break
        else:
            a+=1







def btn_clicked():
    print("Button Clicked")


window = Tk()

window.geometry("1229x693")
window.configure(bg = "#ffffff")
canvas = Canvas(
    window,
    bg = "#ffffff",
    height = 693,
    width = 1229,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge")
canvas.place(x = 0, y = 0)

entry0_img = PhotoImage(file = f"img_textBox0.png")
entry0_bg = canvas.create_image(
    997.5, 199.0,
    image = entry0_img)

entry0 = Entry(
    bd = 0,
    bg = "#efeaea",
    highlightthickness = 0)

entry0.place(
    x = 849.0, y = 169,
    width = 297.0,
    height = 58)

entry1_img = PhotoImage(file = f"img_textBox1.png")
entry1_bg = canvas.create_image(
    997.5, 400.0,
    image = entry1_img)

entry1 = Entry(
    bd = 0,
    bg = "#efeaea",
    highlightthickness = 0)

entry1.place(
    x = 849.0, y = 370,
    width = 297.0,
    height = 58)

entry2_img = PhotoImage(file = f"img_textBox2.png")
entry2_bg = canvas.create_image(
    253.5, 505.0,
    image = entry2_img)

entry2 = Entry(
    bd = 0,
    bg = "#efeaea",
    highlightthickness = 0)

entry2.place(
    x = 162.0, y = 482,
    width = 183.0,
    height = 44)

background_img = PhotoImage(file = f"background.png")
background = canvas.create_image(
    402.5, 346.5,
    image=background_img)

img0 = PhotoImage(file = f"img0.png")
b0 = Button(
    image = img0,
    borderwidth = 0,
    highlightthickness = 0,
    command = btn_clicked,
    relief = "flat")

b0.place(
    x = 915, y = 265,
    width = 158,
    height = 30)

lbl = ttk.Label(window, text = "Enter the Word:").grid(column = 0, row = 0)# Click event  





def click():   
    print(name.get())# Textbox widget  
    search_paper(name.get())
    print("input : " , name.get())# Textbox widget
    print("processed")
name = tk.StringVar()  
nameEntered = ttk.Entry(window, width = 12, textvariable = name).grid(column = 0, row = 1)# Button widget  
button = ttk.Button(window, text = "submit", command = click).grid(column = 1, row = 1)  

window.resizable(False, False)
window.mainloop()
