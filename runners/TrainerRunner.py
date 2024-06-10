import cv2
from vision.CardModelTrainer import CardModelTrainer
from vision.CardRecognizer import CardRecognizer
from model.Board import Board
from model.Computations import Computations
from vision.LabelCards import label_and_save
import os


CSV_PATH =  'resources/labels.csv'
class TrainerRunner:

    def main():
        # First, train the models using CardModelTrainer
        trainer = CardModelTrainer(CSV_PATH)
        trainer.train_models()


if __name__ == "__main__":
    TrainerRunner.main()
