from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
# secure_filename is used to sanitize and secure filename before storing it
from werkzeug.utils import secure_filename
import os
import datetime
# To decode the base64 data URL to obtain the image data
import base64
from PIL import Image
from io import BytesIO

import cv2
from vision.CardModelTrainer import CardModelTrainer
from vision.CardRecognizer import CardRecognizer
from model.Board import Board
from model.Computations import Computations
from vision.LabelCards import label_and_save
import sys
import argparse


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'image')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


ALLOWED_EXTENSIONS = ['jpg','png','jpeg']
app.config['IMAGE_SIZE'] = (500, 500)  # Define the size to which you want to resize the image

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    filename = ''
    error_message = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = 'No file part'
            return render_template('upload.html', filename=filename, error_message=error_message)
        file = request.files['file']
        if file.filename == '':
            error_message = 'No selected file'
            return render_template('upload.html', filename=filename, error_message=error_message)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            #modifying the image (resizing etc) makes the rectangles weird! why! 
            #I dont understand because shouldnt it treat the modified image just like any other??
            #ALSO the classification seems to be WORSE when the image is in its original size!!!



            # Open the uploaded image file
            img = Image.open(file)
            # Resize the image
            img = img.resize(app.config['IMAGE_SIZE'])
            # Save the resized image
            img.save(file_path)

            print(file_path)
            highlighted_sets = getHighlightedBoard(file_path)
            cv2.imwrite(file_path, highlighted_sets)
            
            return render_template('upload.html', filename=filename, error_message=error_message)
        else:
            error_message = 'Allowed file types are png, jpg, jpeg'
    return render_template('upload.html', filename=filename, error_message=error_message)


#capture route to capture image from camera, save it to UPLOAD_FOLDER and then render it on the same page
@app.route('/capture' , methods=['GET','POST'] )
def capture():
    filename = ''  # using filename variable to display video feed and captured image alternatively on the same page
    image_data_url = request.form.get('image')
    if request.method == 'POST' and image_data_url:
        # Decode the base64 data URL to obtain the image data
        image_data = base64.b64decode(image_data_url.split(',')[1])
        # Create an image from the decoded data
        img = Image.open(BytesIO(image_data))
        # Generate a filename with the current date and time
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = f"img_{timestamp}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(file_path, 'PNG')

        highlighted_sets = getHighlightedBoard(file_path)
        cv2.imwrite(file_path, highlighted_sets)

        return render_template('capture.html', filename=filename)
    return render_template('capture.html', filename=filename)



def getHighlightedBoard(file_path):
    recognizer = CardRecognizer()
    image = cv2.imread(file_path)

    cards_and_bounds = recognizer.detect_cards(image)
    card_dict = recognizer.classify_detected_cards(cards_and_bounds, False)
    board = Board()
    for card in card_dict.keys():
        board.addCard(card)
    sets = Computations.getAllSets(board)
    highlighted_sets = recognizer.highlight_sets(image, card_dict, sets)
    return highlighted_sets

if __name__ == '__main__':
    context = ('cert.pem', 'key.pem')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    #app.run(host='0.0.0.0', port=5050, debug=True)
    app.run(debug=False, host='0.0.0.0', ssl_context=context)

    # app.run(debug=True)