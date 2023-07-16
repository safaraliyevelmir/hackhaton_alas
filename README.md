# Alas Hackhaton Losers Team Project 

This project is a web application that utilizes FastAPI to provide predictions from three different models. The application takes input from an HTML form and submits it to the server for processing. The server then uses the three models to generate predictions based on the submitted data.

## Phone Price Predict Model

- Logistic Regression model
- Accuracy: 0.955

## Credit Card Classification Model

- GradientBoostingClassifier model
- Accuracy: 0.962


## Find Number Nueral network

- Tensorflow VGG16 model
- Accuracy: 0.92

## Getting Started

These instructions will guide you to run and develop the project on your local machine.

### Prerequisites

The project requires the following software to be installed:

- Python (version 3.7 or above)
- Pip (Python package manager)

### Installation

1. Download or clone the project files:

```bash
git clone https://github.com/safaraliyevelmir/hackhaton_alas.git
```

2. Create Virtual Enviorement

```bash
python3 -m venv .venv
```

3. Install packages

```bash
pip install -r requirements.txt
```

4. Run Fast api

```bash
uvicorn main:app --reload
```

5. Open Html File

