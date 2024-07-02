# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import json

app = Flask(__name__)

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./smart_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_smart_scores(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    score = probs.detach().numpy()[0][1]  # Get the probability of the positive class
    return score

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    if not data or 'elements' not in data:
        return jsonify({"error": "Invalid input, 'elements' key is required"}), 400

    elements = data['elements']

    results = {}
    for key, text in elements.items():
        if isinstance(text, dict):
            # If the value is a dictionary, evaluate each nested value
            nested_results = {}
            for sub_key, sub_text in text.items():
                criteria = {
                    "Specific": get_smart_scores(f"Is this goal specific? {sub_text}"),
                    "Measurable": get_smart_scores(f"Is this goal measurable? {sub_text}"),
                    "Achievable": get_smart_scores(f"Is this goal achievable? {sub_text}"),        
                    "Time-bound": get_smart_scores(f"Is this goal time-bound? {sub_text}")
                }

                # Convert all numpy.float32 values to Python float
                criteria = {k: float(v) for k, v in criteria.items()}

                total_score = sum(criteria.values())
                average_score = total_score / len(criteria)

                # Check if the average score meets the threshold
                threshold = 0.5
                if average_score > threshold:
                    evaluation_message = "This goal meets the SMART criteria!"
                else:
                    evaluation_message = "This goal does not fully meet the SMART criteria."
                    for criterion, score in criteria.items():
                        if score <= threshold:
                            evaluation_message += f" The goal is not {criterion.lower()}."

                # Add the average score and evaluation message to the response dictionary
                nested_results[sub_key] = {
                    "text": sub_text,
                    "criteria": criteria,
                    "average_score": average_score,
                    "evaluation_message": evaluation_message
                }
            results[key] = nested_results
        else:
            # If the value is not a dictionary, evaluate it directly
            criteria = {
                "Specific": get_smart_scores(f"Is this goal specific? {text}"),
                "Measurable": get_smart_scores(f"Is this goal measurable? {text}"),
                "Achievable": get_smart_scores(f"Is this goal achievable? {text}"),        
                "Time-bound": get_smart_scores(f"Is this goal time-bound? {text}")
            }

            # Convert all numpy.float32 values to Python float
            criteria = {k: float(v) for k, v in criteria.items()}

            total_score = sum(criteria.values())
            average_score = total_score / len(criteria)

            # Check if the average score meets the threshold
            threshold = 0.5
            if average_score > threshold:
                evaluation_message = "This goal meets the SMART criteria!"
            else:
                evaluation_message = "This goal does not fully meet the SMART criteria."
                for criterion, score in criteria.items():
                    if score <= threshold:
                        evaluation_message += f" The goal is not {criterion.lower()}."

            # Add the average score and evaluation message to the response dictionary
            results[key] = {
                "text": text,
                "criteria": criteria,
                "average_score": average_score,
                "evaluation_message": evaluation_message
            }

    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
