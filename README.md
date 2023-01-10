# Meme-Sentimental-Analysis
This model is trained on total of 6992 images with 30/70 split ratio
between test data and train data. 
This model classifies memes as either Positive(1), Negative(-1), Neutral(0).
6 different machine learning models from sklearn library are trained, 3 for images and 3 for text.
To get single output, majority voting is done between all 6 outcomes.
In order to use this model, basic web application is designed using flask, which take single image input 
and it will show either image is positive, negative or neutral.
