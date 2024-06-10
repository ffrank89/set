import cv2
from vision.CardRecognizer import CardRecognizer
from model.Board import Board
from model.Computations import Computations
import argparse

DEFAULT_PATH = 'resources/board_photos/Screenshot 2024-05-14 at 5.06.17 PM.jpg'


class RecognizerRunner:

    def main(image_path=DEFAULT_PATH):
        recognizer = CardRecognizer()
        image = cv2.imread(image_path)
        cv2.imshow(f'Card', image)
        cv2.waitKey(0)


    
        cards_and_bounds = recognizer.detect_cards(image)
        print("Recognized ", len(cards_and_bounds), " cards.")
        card_dict = recognizer.classify_detected_cards(cards_and_bounds, False)
        board = Board()
        for card in card_dict.keys():
            board.addCard(card)

        sets = Computations.getAllSets(board)

        highlighted_sets = recognizer.highlight_sets(image, card_dict, sets)
        #highlighted_sets = recognizer.highlight_board(image, card_dict)
        cv2.imshow("Highlighted Board", highlighted_sets)
        cv2.imwrite('static/image/gat.png', highlighted_sets)

        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', nargs='?', default=DEFAULT_PATH)
    args = parser.parse_args()
    RecognizerRunner.main(args.image_path)


