import os
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

app = Flask(__name__)
CORS(app)

# Inicializar variables globales
tokenizer = None
model = None
dataset = None

def init_model():
    global tokenizer, model, dataset
    try:
        # Cargar el modelo y tokenizer desde Hugging Face
        MODEL_PATH = "hobbycard/puntacana-bot"
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Cargar el dataset
        with open('condostel.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        return True
    except Exception as e:
        print(f"Error inicializando el modelo: {e}")
        return False

def calculate_model_score(question, candidate_question):
    try:
        inputs = tokenizer(
            candidate_question,
            question,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits.numpy()[0][0]
            
        return score
    except Exception as e:
        print(f"Error al calcular score del modelo: {e}")
        return -float('inf')

@app.route('/')
def home():
    return "PuntaCana Bot API está funcionando!"

@app.route('/api/chat', methods=['POST'])
def chat():
    global tokenizer, model, dataset
    
    # Inicializar el modelo si no está cargado
    if tokenizer is None or model is None or dataset is None:
        if not init_model():
            return jsonify({
                'error': 'Error inicializando el modelo',
                'status': 'error'
            }), 500
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonif