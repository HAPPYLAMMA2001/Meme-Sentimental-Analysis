from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import pytesseract
import string
from textblob import TextBlob
import nltk.corpus
from nltk.corpus import stopwords
import nltk
import re
from skimage.transform import resize
from skimage.io import imread
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import ImageFile
from skimage import feature

ImageFile.LOAD_TRUNCATED_IMAGES = True


with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\sgdc_text.pkl", 'rb') as file:  #SGDClassifier
    sgdc = pickle.load(file)

with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\random_forest_text.pkl", 'rb') as file:  #random forest classifier
    rf = pickle.load(file)

with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\dtc_text.pkl", 'rb') as file:  #Decision tree classifier
    dtc = pickle.load(file)

with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\lr_img.pkl", 'rb') as file:  #logistic regression
    lr = pickle.load(file)

with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\mnb_img.pkl", 'rb') as file:  #MultinomialNB
    mnb = pickle.load(file)

with open(r"C:\Users\shami\Desktop\University\Intro to DS\models\gnb_img.pkl", 'rb') as file:  #GaussianNB
    gnb = pickle.load(file)



pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def clean(x):
    stop = stopwords.words('english')
    x = x.lower()
    tokens = nltk.word_tokenize(x)
    removed = []
    for i in tokens:
        if i.startswith('@') == False and i.endswith('.com') == False:
            removed.append(i)
    y = ' '.join(removed)
    combine = []
    y = y.translate(str.maketrans('','',string.punctuation)) # removing punctuations

    y = nltk.word_tokenize(y)
    used = set()
    clear = []
    for words in y:
        if words not in used:
            used.add(words)
            clear.append(words)
    y = ' '.join(clear)

    y = ''.join([i for i in y if not i.isdigit()])
    y = " ".join(y.split())

    y = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", y) #removing links and emojis nd ascii

    corr = TextBlob(y)
    y = corr.correct()

    keep = []
    for w in y.split():
        if w not in stop:
            keep.append(w)
    y = " ".join(keep)
    return y

def textExtraction(path):
    text=pytesseract.image_to_string(path)
    text=clean(text)
    return text


raw = pd.read_csv(r"C:\Users\shami\Desktop\University\Intro to DS\models\working\finale_text.csv",usecols = ['text_corrected'])

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def upload():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def answer():
    img = request.files['img']
    if (img != ""):
        imgpath = './images/' + img.filename
        img.save(imgpath)
        txt = textExtraction(imgpath)
        text = raw.copy(deep=True)
        text.loc[len(text.index)] = txt
        vectorizer = TfidfVectorizer(lowercase=True,max_features=9978)
        X = vectorizer.fit_transform(text['text_corrected'])
        temp = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
        test = [temp.iloc[len(temp.index.values)-1]] #test value for X


        x1 = imread(imgpath,as_gray=True)
        x1 = resize(x1, (200,200))
        x1 = feature.canny(x1)
        x1= x1.astype('int32')
        x1 = [list(np.concatenate(x1).flat)]

        pred = [sgdc.predict(test)[0],rf.predict(test)[0],dtc.predict(test)[0],lr.predict(x1)[0],mnb.predict(x1)[0],gnb.predict(x1)[0]]
    
        pos,neg,neu = 0,0,0
        for i in range(6):
            if pred[i] == 1:
                pos += 1
            elif pred[i] == 0:
                neu += 1
            else:
                neg += 1

        if (pos > neg) and (pos > neu):
            ans = "POSITIVE"
        elif (neg > pos) and (neg > neu):
            ans = "NEGATIVE"
        else:
            ans = "NEUTRAL"


    print(pred)
        
    
    return render_template('index.html', answer = ans)

if __name__ == '__main__':
    app.run(port=3000,debug=True)