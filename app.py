import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "PuntaCana Bot API está funcionando!"

# Inicializar variables globales
tokenizer = None
model = None
dataset = None

def init_model():
    global tokenizer, model, dataset
    try:
        # Cargar el modelo y tokenizer desde Hugging Face
        MODEL_PATH = "hobbycard/p
    
   
