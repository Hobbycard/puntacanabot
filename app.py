from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json
import os

app = Flask(__name__)
CORS(app)

# Cargar el modelo y tokenizer desde Hugging Face
MODEL_PATH = "hobbycard/puntacana-bot"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

# Cargar el dataset
with open('condostel.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

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
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No se proporcionó ninguna pregunta'}), 400
        
        # Buscar la mejor respuesta
        mejor_score = -float('inf')
        mejor_respuesta = None
        
        for item in dataset:
            score = calculate_model_score(question, item["question"])
            if score > mejor_score:
                mejor_score = score
                mejor_respuesta = item["answer"]

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
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)