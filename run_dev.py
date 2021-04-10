# flask libraries
from flask import Flask,render_template, redirect, url_for, request, session,flash,get_flashed_messages,jsonify
from flask_login import login_required
from functools import wraps
import os
from os import listdir
from os.path import isfile, join
import boto3

s3 = boto3.resource('s3')
s3_bucket_name = 'buckeyname'
## NN libraries (change based on what we end up using)
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

### laoding teh models
json_file = open('model/model.json','r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = tf.keras.models.model_from_json(loaded_model_json)

loaded_model.load_weights("model/model.h5")

def find_class(file_name):
    img = image.load_img(file_name,target_size=(200,200))
    X= image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = loaded_model.predict(images)
    if val ==0:
        return('bycle')
    else:
        return('car')
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
        # s3.meta.client.upload_file(f'static/uploads/{b}', s3_bucket_name, b)
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
