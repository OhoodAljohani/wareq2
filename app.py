# loading the model and preparing for prediction 
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)
print(keras.__version__)
import numpy as np
import tensorflow as tf
import pandas as pd 
import s3fs
import zipfile
import tempfile
from pathlib import Path
import logging


AWS_ACCESS_KEY="AKIAVN6Q233WCVBSNNYB"
AWS_SECRET_KEY="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"
BUCKET_NAME="elasticbeanstalk-us-east-1-373563252460"
def get_s3fs():
  return s3fs.S3FileSystem(key=AWS_ACCESS_KEY, secret=AWS_SECRET_KEY)

def s3_get_keras_model(model_name: str) -> keras.Model:
  with tempfile.TemporaryDirectory() as tempdir:
    s3fs = get_s3fs()
    # Fetch and save the zip file to the temporary directory
    s3fs.get(f"{BUCKET_NAME}/{model_name}.zip", f"{tempdir}/{model_name}.zip")
    # Extract the model zip file within the temporary directory
    with zipfile.ZipFile(f"{tempdir}/{model_name}.zip") as zip_ref:
        zip_ref.extractall(f"{tempdir}/{model_name}")
    # Load the keras model from the temporary directory
    return keras.models.load_model(f"{tempdir}/{model_name}")
model = s3_get_keras_model("mymodel97")
#model = keras.models.load_model("/model/mymodel97.h5")
from flask import Flask, render_template,request
import os 
app = Flask(__name__)

clases_names = pd.read_csv("Data/classes.csv",index_col=[0])
#clases_names.info()
classes_names = clases_names.sort_values(by="num",inplace=True)
#clases_names.head()
clases_names.index = range(1, 305, 1)
#clases_names.to_csv("Data/data.csv",index=False)
#print(clases_names.iloc[0,1])
app = Flask(__name__)


model.make_predict_function()
import numpy as np
def predict_label(img_path):
    image_size = (224,224)
#model.summary()
# #Predict model
    img = tf.keras.preprocessing.image.load_img(
    img_path, target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)/255
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    output=np.argmax(predictions,axis=1)
    classes = {'0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6,\
    '15': 7, '16': 8, '17': 9, '18': 10, '19': 11, '2': 12, '20': 13,\
    '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, '26': 19,\
    '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25,\
    '32': 26, '33': 27, '34': 28, '35': 29, '36': 30, '37': 31,\
    '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, '42': 37,\
    '43': 38, '44': 39, '45': 40, '46': 41, '47': 42, '48': 43,\
    '49': 44, '5': 45, '50': 46, '51': 47, '52': 48,\
    '53': 49, '54': 50, '55': 51, '56': 52,\
    '57': 53, '58': 54, '59': 55,\
    '6': 56, '60': 57, '61': 58, '62': 59, \
    '63': 60, '64': 61, '65': 62, '66': 63,\
    '67': 64, '68': 65, '69': 66, '7': 67,\
    '70': 68, '71': 69, '72': 70, '73': 71, '74': 72,\
    '75': 73, '76': 74, '77': 75, '78': 76, '79': 77, '8': 78,\
    '80': 79, '81': 80, '82': 81, '83': 82, '84': 83, '85': 84,\
    '86': 85, '87': 86, '88': 87, '89': 88, '9': 89, '90': 90, '91': 91,\
    '92': 92, '93': 93, '94': 94, '95': 95, '96': 96, '97': 97, '98': 98, '99': 99}
    classes = dict((v,k) for k,v in classes.items())
    result = np.array_str(output)
    #print(type(result))
    r = result.lstrip("[").rstrip("]")
    result = int(r)
    o = int(classes[result])
    d = pd.read_csv("data/data.csv")
    f = pd.read_csv("data/classes_info.csv",index_col=[0])
    r = d.iloc[o,1]
    g = f.iloc[o,0]
    c = f.iloc[o,1]
    t = f.iloc[o,2]
    #print("The images is predicted : ",r)
    #print(
    #    "This image is predicted as ",output
    #)
    return r ,g,c,t



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("main.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p ,g,c,t= predict_label(img_path)

	return render_template("main.html", prediction = p, g=g,t=t,c=c,img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	#app.run(debug = True)
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=True)
