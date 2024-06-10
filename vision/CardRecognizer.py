import cv2
import joblib
from model.Shape import Shape, ShapeType
from model.Color import Color, ColorType
from model.Number import Number
from model.Shading import Shading, ShadingType
from model.Card import Card
from keras._tf_keras.keras.models import load_model
import random
import numpy as np
from enum import Enum
from collections import defaultdict



class RGBColor(Enum):
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    HOT_PINK = (255, 0, 255)
    PURPLE = (128, 0, 128)
    LIME = (0, 255, 0)
    LIGHT_PURPLE = (255, 192, 203)
    GREY = (128, 128, 128)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

class CardRecognizer:
    def __init__(self):
        self.shape_model = load_model('models/shape_model_fine_tuned.keras')
        self.color_model = load_model('models/color_model_fine_tuned.keras')
        self.number_model = load_model('models/number_model_fine_tuned.keras')
        self.shading_model = load_model('models/shading_model_fine_tuned.keras')

    def preprocess_image(self, image):
        resized = cv2.resize(image, (128, 128))  # Resize to a fixed size
        normalized = resized / 255.0  # Normalize pixel values
        return normalized
    
    def detect_cards(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards_and_bounds = []
        for i, contour in enumerate(contours):
            approx = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.contourArea(contour) > 2500:
                print(cv2.contourArea(contour))
                x, y, w, h = cv2.boundingRect(approx)
                card = image[y:y + h, x:x + w]                

                cards_and_bounds.append((card, [x,y,w,h]))
                

        return cards_and_bounds

    def classify_detected_cards(self, cards_and_bounds, display=False):
        print("Amount of cards: ", len(cards_and_bounds))
        card_dict = {}
        i = 0
        for card_image, box in cards_and_bounds:
            i+=1
            card = self.create_card(card_image)
            card_dict[card] = box
            if display:
                cv2.imshow(f'Card {i+1}', card_image)
                cv2.waitKey(0)
        return card_dict
    
    def classify_shape(self, image):
        preprocessed_image = self.preprocess_image(image)
        feature_vector = np.expand_dims(preprocessed_image, axis=0)  # Reshape for the classifier
        prediction = self.shape_model.predict(feature_vector)
        shape_class = np.argmax(prediction)
        return Shape(ShapeType(shape_class))

    def classify_color(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        hist.flatten()
        hist = np.expand_dims(hist, axis=0)  # Reshape for the classifier
        prediction = self.color_model.predict(hist)
        color_class = np.argmax(prediction)
        return Color(ColorType(color_class))

    def classify_number(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count the number of significant contours
        count = 0
        for contour in contours:
            if cv2.contourArea(contour) > 4500:
                count += 1

        # The number of shapes should correspond to the number
        if count == 1:
            return Number(1)
        elif count == 2:
            return Number(2)
        elif count == 3:
            return Number(3)
        else:
            # Fallback to model prediction if contour count is ambiguous
            preprocessed_image = self.preprocess_image(image)
            feature_vector = np.expand_dims(preprocessed_image, axis=0)
            prediction = self.number_model.predict(feature_vector)
            number_class = np.argmax(prediction)
            return Number(number_class)

    def classify_shading(self, image):
        preprocessed_image = self.preprocess_image(image)
        feature_vector = np.expand_dims(preprocessed_image, axis=0)
        prediction = self.shading_model.predict(feature_vector)
        shading_class = np.argmax(prediction)
        return Shading(ShadingType(shading_class))

    def create_card(self, image):
        shape = self.classify_shape(image)
        color = self.classify_color(image)
        number = self.classify_number(image)
        shading = self.classify_shading(image)
        return Card(shape, color, number, shading)
    
    def highlight_sets(self, image, card_dict, sets):


        set_count = defaultdict(int)
        for card_set in sets:
            for card in card_set:
                set_count[card] += 1
        
        offset = 25
        
        colors = list(RGBColor)
        for i, card_set in enumerate(sets):
            color = colors[i % len(colors)]
            print("============")
            print(color.name, i+1)
            print("------------------")

            for card in card_set:
                print(card)
                print(".............")
                x,y,w,h = card_dict[card]
                cv2.rectangle(image, (x + (offset * set_count[card]), y + (offset * set_count[card])), 
                              (x + w, y + h), color.value, 10)
                set_count[card] -= 1

        return image
    

    def highlight_board(self, image, card_dict):
        
        colors = list(RGBColor)
        print(len(card_dict))
        for i, card in enumerate(card_dict.keys()):
            color = colors[i % len(colors)]
            print(color, i+1)
            print("=============")
            print(card)
            print("------------------")
            x,y,w,h = card_dict[card]

            cv2.rectangle(image, (x, y), (x + w, y + h), color.value, 15)
            cv2.putText(image, i+1, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, 10, 5)


        return image