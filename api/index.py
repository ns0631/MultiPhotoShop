from flask import Flask, render_template, request
import numpy as np
import cv2, os, sys, glob, json

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/processed', methods = ['POST'])   
def success():   
    if request.method == 'POST':
        print(request.files)
        src = request.files['srcimage'] 
        dest = request.files['destimage']
        if src is None or dest is None:
            pass
        else:
            srcstream = src.read()
            deststream = dest.read()
            
            srcimg = cv2.imdecode(np.fromstring(srcstream,np.uint8), cv2.IMREAD_COLOR)
            destimg = cv2.imdecode(np.fromstring(deststream,np.uint8), cv2.IMREAD_COLOR)

        return render_template("acknowledgement.html") 

@app.route('/about')
def about():
    return 'About'