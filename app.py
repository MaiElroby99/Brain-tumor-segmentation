
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from main import getPrediction
from PIL import Image
import os
import numpy as np
import random
import glob


#Save images to the 'static' folder as Flask serves images from this directory
UPLOAD_FOLDER = 'static/images/'

#Create an app object using the Flask class. 
app = Flask(__name__, static_folder="static")

#Add reference fingerprint. 
#Cookies travel with a signature that they claim to be legit. 
#Legitimacy here means that the signature was issued by the owner of the cookie.
#Others cannot change this cookie as it needs the secret key. 
#It's used as the key to encrypt the session - which can be stored in a cookie.
#Cookies should be encrypted if they contain potentially sensitive information.
app.secret_key = "secret key"

#Define the upload folder to save images uploaded by the user. 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Define the route to be home. 
#The decorator below links the relative route of the URL to the function it is decorating.
#Here, index function is with '/', our root directory. 
#Running the app sends us to index.html.
#Note that render_template means it looks for the file in the templates folder. 
@app.route('/predict-tumor-segmentation')

def index():
    return render_template('index2.html')



#Add Post method to the decorator to allow for form submission. 
@app.route('/predict-tumor-segmentation', methods=['POST' , 'GET'])
def submit_file():
    files = glob.glob('static/images/*')
    for f in files:
      os.remove(f)
      
    rnd = random.randint(0,999999)
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)  #Use this werkzeug method to secure filename. 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            pred  = getPrediction(filename)
            full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            flash(full_filename)
            im = Image.fromarray(np.squeeze((pred* 255).astype(np.uint8)))
            # im = im.convert('RGB')
            # if i not in numbers:
            im.save(f"static/images/prediction{rnd}.png")
            image_path = f'prediction{rnd}.png'
            image_pred = os.path.join('static/images/', image_path)
            flash(image_pred)
            return redirect('/predict-tumor-segmentation')


if __name__ == "__main__":
    app.run(debug=True , port=8080)
    
    
    
    
    
    