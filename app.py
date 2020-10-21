from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import xgboost as xgb
import pickle
import re
import base64

import sys
import os
sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)
global model
model = pickle.load(open('model/xgb_reg.pkl', "rb"))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
    # x = np.invert(x)
    x = imresize(x, (28, 28, 1))

    x = np.array(x).reshape((1, -1))
    x = xgb.DMatrix(x)

    out = int(model.predict(x))
    print(out)

    response = str(out)
    return response


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
