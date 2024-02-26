import torch

from flask import Flask, render_template, request
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

from test import get_metadata

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index', methods = ['POST'])
def get_image():
    test = request.form.get('img')

    # if not bool(url.strip()):
    # url="http://images.cocodataset.org/val2017/000000039769.jpg"
    
    data = get_metadata(test)

    return render_template("index.html",
                           title = "test",
                           data = data,
                           image = test,
                           test = test,
                           )