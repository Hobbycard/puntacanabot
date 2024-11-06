import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json

# Configuración del cache de transformers
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers'

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "PuntaCana Bot API está funcionando!"

# Variables globales
tokenizer = None
model = None
dataset = None

# Función para inicializar el modelo y el dataset
def init_model():
    global tokenizer, model, dataset
    try:
        # Definir ruta y token de autenticación
        MODEL_PATH = "hobbycard/puntacana-bot"  # Ruta del modelo en Hugging Face
        auth_token = os.environ.get('HUGGINGFACE_TOKEN')  # Token de autenticación
        
        # Cargar tokenizer y modelo
        tokenizer = RobertaTokenizer.from_pretrained(
            MODEL_PATH,
            use_auth_token=auth_token
        )
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_PATH,
            use_auth_token=auth_token
        )
        
        # Cargar el dataset
        with open('condostel.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        return True
    except Exception as e:
        print(f"Error inicializando el modelo: {e}")
        return False

# Función para calcular el score del modelo
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

@app.route('/api/chat', methods=['POST'])
def chat():
    global tokenizer, model, dataset
    
    try:
        # Inicializar el modelo si no está cargado
        if tokenizer is None or model is None or dataset is None:
            if not init_model():
                return jsonify({
                    'error': 'Error inicializando el modelo',
                    'status': 'error'
                }), 500
        
        # Obtener pregunta desde la solicitud
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No se proporcionó ninguna pregunta'}), 400
        
        # Buscar la mejor respuesta en el dataset
        mejor_score = -float('inf')
        mejor_respuesta = None
        
        for item in dataset:
            score = calculate_model_score(question, item["question"])
            if score > mejor_score:
                mejor_score = score
                mejor_respuesta = item["answer"]

        # Enviar respuesta basada en el score obtenido
        if mejor_score > 0.8:
            return jsonify({
                'answer': mejor_respuesta,
                'status': 'success'
            })
        else:
            return jsonify({
                'answer': "Lo siento, no tengo información específica sobre eso. ¿Podrías reformular tu pregunta?",
                'status': 'no_match'
            })

    except Exception as e:
        print(f"Error en chat: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
