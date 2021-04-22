# flask libraries
from flask import Flask,render_template, redirect, url_for, request, session,flash,get_flashed_messages,jsonify
from flask_login import login_required
from functools import wraps
import os
from os import listdir
from os.path import isfile, join
import boto3

s3 = boto3.resource('s3')
s3_bucket_name = 'group6clouddeeplearning'
## NN libraries (change based on what we end up using)
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import cv2
#
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.optimizers import RMSprop

from PIL import Image
from torchvision import transforms
import pickle
import torch

### laoding teh models
#unpickle

model = pickle.load(open('model/resnet_pickle.pkl', "rb"))

def find_class(file_name,model = model): #Feed in the image
    class_names = ['Normal', 'Viral', 'COVID-19']
    image = Image.open(file_name)
    img_xray = image.convert('RGB')
    preprocess = transforms.Compose([
              transforms.Resize(256),
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize(
              mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]
          )])
    img_xray_preprocessed = preprocess(img_xray)
    batch_image_xray_tensor = torch.unsqueeze(img_xray_preprocessed, 0)
    model.eval()
    out = model(batch_image_xray_tensor)
    _, preds = torch.max(out, 1)
    return class_names[int(preds)]
#############################
######### FLASK APP #########

app = Flask(__name__,static_url_path='/static')
app.secret_key = 'Secret key here can be anything'
# app_root = os.path.dirname(os.path.abspath(__file__))

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# function to require login for othe rpages
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('login'))

    return wrap

@app.route('/homepage')
@login_required
def homepage():
    return render_template('home.html')

@app.route('/upload_page',methods = ['GET', 'POST'])
@login_required
def upload_page():
    return render_template('upload.html')

@app.route('/truth_page_link',methods = ['GET', 'POST'])
@login_required
def truth_page_link():
    onlyfiles0 = [f for f in listdir('static/uploads') if isfile(join('static/uploads', f))]
    onlyfiles = ['Select The image'] + onlyfiles0
    s3_folder = ['covid','normal']
    a = request.form.get('s3_folder')
    b = request.form.get('onlyfilehtml')
    if (a == 'class') | (b == 'Select The image') | (a == None) |(b ==None) :
        # flash('Select the files to be ')
        message = 'Please select the file to be uploaded'
        return render_template('ground_truth.html', onlyfiles=onlyfiles,message=message)
    # elif (b != 'Class') & (a != 'Select The image'):
    elif (b != 'Select The image'):
        s3.meta.client.upload_file(f'static/uploads/{b}', s3_bucket_name,f'{a}/{b}')
        os.remove(f'static/uploads/{b}')
        message = (f'{b} was successfully uplaoded to S3 {a} folder')
    return render_template('ground_truth.html', onlyfiles=onlyfiles,message=message)

@app.route('/posted')
def posted():
    return render_template('posted.html')


#login page using user and password
@app.route('/',methods = ['GET','POST'])
def login():
    error =None
    if request.method == 'POST':
        if request.form['username'] != 'user' or request.form['password'] != 'pass':
            error = 'Invalid credentials. Please try again.'
        else:
            session['logged_in'] = True
            # flash('You are Logged in')
            return redirect(url_for('upload_page'))
    return render_template('login.html',error = error)

@app.route('/logout')
def logout():
    session.pop('logged_in',None)
    flash('You are Logged out')
    return redirect(url_for('login'))

@app.route('/upload_form')
@login_required
def upload_form():
	return render_template('upload.html')

@app.route('/upload',methods = ["POST","GET"])
@login_required
def upload():
    target = 'static/uploads'
    print(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = f"{target}/{filename}"
        print(destination)
        file.save(destination)
    model_res = find_class(destination)
    return render_template('result.html',user_image = destination,val_res=model_res)

if __name__ == '__main__':
    app.run()
