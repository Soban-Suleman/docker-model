# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 18:59:59 2024

@author: HP
"""

from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import numpy as np
import re
import json
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the model and tokenizer for SMART evaluation
smart_model = BertForSequenceClassification.from_pretrained('./smart_model')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model for similarity analysis
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_smart_scores(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = smart_model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    score = probs.detach().numpy()[0][1]  # Get the probability of the positive class
    return score

def preprocess_text(text):
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def normalize_weights(weights):
    total = sum(weights)
    return [w / total for w in weights]

def compute_similarity_scores(project, business, weights, threshold=0.50):
    if len(weights) != len(business):
        raise ValueError("The number of weights must match the number of business statements.")
    
    all_scores = []
    project_embeddings = similarity_model.encode(list(project.values()), convert_to_tensor=True)
    business_embeddings = similarity_model.encode(list(business.values()), convert_to_tensor=True)
    
    project_keys = list(project.keys())
    business_keys = list(business.keys())
    
    for i, p_embedding in enumerate(project_embeddings):
        scores = []
        applied_weights = []
        for j, b_embedding in enumerate(business_embeddings):
            score = util.pytorch_cos_sim(p_embedding, b_embedding).item()
            print(f"Similarity between '{project_keys[i]}' and '{business_keys[j]}': {score:.4f}")
            if score >= threshold:
                scores.append(score)
                applied_weights.append(weights[j])
        
        if scores:
            total_weight = sum(applied_weights)
            weighted_avg_score = sum(score * weight for score, weight in zip(scores, applied_weights)) / total_weight
        else:
            weighted_avg_score = 0
        
        all_scores.append({
            'Project Statement': project_keys[i],
            'Weighted Average Similarity Score': weighted_avg_score
        })
    
    return pd.DataFrame(all_scores)

def normalize_single_value(value, max_value, min_value=0):
    if max_value == min_value:
        return 0.5
    return (value - min_value) / (max_value - min_value)

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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    artifacts = data['artifacts']
    
    # Initialize empty dictionaries
    project_data = None
    business_data = None
    
    # Determine which artifact is the project and which is the business plan
    for artifact in artifacts:
        if artifact['artifactName'] == 'Charter':
            project_data = artifact['GROKS']
        elif artifact['artifactName'] == 'Business Plan':
            business_data = artifact['GROKS']
    
    if project_data is None or business_data is None:
        return jsonify({"error": "Both 'Charter' and 'Business Plan' artifacts must be provided."}), 400
    
    project_statements = {}
    business_statements = {}
    
    for key, values in project_data.items():
        for sub_key, value in values.items():
            project_statements[f"{key} - {sub_key}"] = preprocess_text(value)
    
    for key, values in business_data.items():
        for sub_key, value in values.items():
            business_statements[f"{key} - {sub_key}"] = preprocess_text(value)
    
    num_business_statements = len(business_statements)
    business_weights = [1.0] * num_business_statements
    normalized_weights = normalize_weights(business_weights)
    
    df_scores = compute_similarity_scores(project_statements, business_statements, normalized_weights)
    similarity_scores = df_scores['Weighted Average Similarity Score'].tolist()
    total_similarity_score = sum(similarity_scores)
    max_value = len(project_statements)
    normalized_total_similarity_score = normalize_single_value(total_similarity_score, max_value=max_value)
    df_scores['Normalized Total Similarity Score'] = normalized_total_similarity_score
    
    # Structure the output in the same format as the input, excluding the Business Plan part
    output = {
        "Total Feasibility Score": normalized_total_similarity_score,
        "artifacts": [
            {
                "artifactName": "Charter",
                "GROKS": {
                    key: {sub_key: {"Value": value, "Similarity Score": df_scores.loc[df_scores['Project Statement'] == f"{key} - {sub_key}", 'Weighted Average Similarity Score'].values[0]}
                          for sub_key, value in values.items()}
                    for key, values in project_data.items()
                }
            }
        ]
    }
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
