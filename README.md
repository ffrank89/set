
# set solver

flask app that uses computer vision and tensorflow keras models to identify and display all of the "sets" in a photo of set cards.

## Features
- Detect and classify individual Set game cards from images
- Highlight valid sets in the uploaded image
- Train custom models for card attributes using TensorFlow and Keras
- Flask web application for easy image upload and processing

## Requirements
- Python 3.x
- [Virtualenv](https://virtualenv.pypa.io/en/stable/) (optional, but recommended)

## Run Locally

Clone the project

```bash
git clone https://github.com/ffrank89/set
```

Go to the project directory

```bash
cd set
```

Install the dependencies
```bash
pip install -r requirements.txt
```

Set up Virtual Environment (optional but will help with dependency conflicts)
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

Train the models

```bash
python3 runners/TrainerRunner.py
```

Run flask app

```bash
python3 app.py
```
