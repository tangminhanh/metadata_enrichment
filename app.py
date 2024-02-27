import torch

from flask import Flask, render_template, request
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

from test import get_metadata, get_metadata_vid

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index', methods = ['POST'])
def get_image():
    url = request.form.get('url')
    select = request.form.get('select')
    
    print(select)
    if select =='image':
        data = get_metadata(url, False)
        display = f'<img src="{url}">'
    else:
        data = get_metadata_vid(url)
        display = f'<video width="320" height="240" controls> <source src="{url}" type="video/mp4"></video>'

    return render_template("index.html",
                           data = data,
                           display = display,
                           test = url,
                           )