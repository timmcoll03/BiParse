from article_parse import open_train_data, parse_article
from neural_net import train_model, load_model
import pickle
import numpy as np
from flask import Flask
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from boilerpipe.extract import Extractor
import random


# to get and format train data, article_parse.get_train_data(), get_unique_in(), format_training_data()

train_freq, train_bias, train_text = open_train_data()


def get_unique_in():

    unique = list()

    total = list()

    for i in train_text:
        print(str(round(100*train_text.index(i)/len(train_text), 2)) + "%", end="\r", flush=True)
        total.extend(i)
    unique = list(set(total))


    with open('tmp/unique_in.pkl', 'wb') as ui:
        pickle.dump(unique, ui)

    print(len(unique))

def load_unique():

    with open('tmp/unique_in.pkl', 'rb') as ui:
        unique = pickle.load(ui)

    return unique

def format_data(data, unique):
    formatted = np.zeros((len(unique)))
    for t in data.keys():
        try:
            formatted[unique.index(t)] = data[t]
        except:
            pass
    return np.asarray([formatted])

def format_training_data():
    unique = load_unique()
    training_data_formatted = list()
    for i, t in enumerate(train_text):
        formatted = np.zeros((len(unique)))
        print(str(round(100*i/len(train_text), 2)) + "%", end='\r', flush=True)
        for i1, t1 in enumerate(t):
            formatted[unique.index(t1)] = train_freq[i][i1]
        training_data_formatted.append(formatted)
    with open('tmp/training_data_formatted.pkl', 'wb') as tdf:
        pickle.dump(training_data_formatted, tdf)

def get_final_training_data():
    with open('tmp/training_data_formatted.pkl', 'rb') as tdf:
        freq = np.asarray(pickle.load(tdf))
    return freq, np.asarray(train_bias)

get_unique_in()
format_training_data()
combined = list(zip(format_training_data, train_bias))
random.shuffle
final_training_freq, final_training_bias = get_final_training_data()
print("Loaded training data, initializing neural net")
train_model(final_training_freq, final_training_bias)

model = load_model()
print(model.summary())

unique = load_unique()

app=Flask(__name__)

@app.route("/")
def get_bias(url):
    # number = articlepredict("articleURLorwhatever")
    # number_msg = "I think that article is {}".format("string")

#if __name__ == '__main__':
#    app.run(debug=True)   

    # FLASK_APP=hello.py flask run
    # page = requests.get(url)
    # def hello():
    #     data = (model.summary())
    #     return render_template('settings.html', data=data)
    
    

    data = parse_article(
        """Conservative politicians and media outlets have championed the cause of Edward Gallagher, a Navy SEALs special operations chief with eight combat and overseas deployments. Trump has tweeted about his case. Gallagher faces some of the most serious charges of those who may receive pardons, including premeditated murder and threatening to kill fellow service members if they informed others of his actions.

Gallagher is set to face a military court and the exact details of his case are not public, but several publications, including The San Diego Union-Tribune and the Navy Times, have obtained court documents and judge’s rulings in the lead-up to the trial that give a substantial account of the charges he faces. """
    )


    data_formatted = format_data(data, unique)

    print(data_formatted)



    print("SIZE")
    print(data_formatted.size)
    return model.predict(data_formatted)


print(get_bias("https://www.cnbc.com/2019/05/25/trump-digs-at-japan-for-substantial-trade-advantage-and-calls-for-more-investment-in-us.html"))