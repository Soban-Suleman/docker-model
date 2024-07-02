# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 01:43:24 2024

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 00:50:12 2024

@author: HP
"""

from flask import Flask, jsonify, request
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer, util

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the dataset from the uploaded Excel file
file_path = 'SMAT Table.xlsx'
df = pd.read_excel(file_path)

# Normalize the titles in the dataframe for comparison
df['Title_normalized'] = df['Title'].str.lower().str.replace(' ', '').str.replace('-', '').str.replace('_', '')

# Load the tokenizer and model for SMART criteria
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./smart_model', ignore_mismatched_sizes=True)

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Load the model for similarity scoring
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Utility functions
def get_goal_scores(goal_title):
    normalized_title = goal_title.lower().replace(' ', '').replace('-', '').replace('_', '')
    goal_scores = df[df['Title_normalized'] == normalized_title]
    if goal_scores.empty:
        return None
    return {
        "specific": float(goal_scores['specific'].values[0]),
        "measurable": float(goal_scores['measurable'].values[0]),
        "achievable": float(goal_scores['achievable'].values[0]),
        "time_bound": float(goal_scores['time_bound'].values[0])
    }

def get_smart_scores(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)
    return probs.detach().numpy()[0]

def evaluate_smart_criteria(title, text):
    scores = get_smart_scores(text)
    criteria = {
        "specific": float(scores[0]),
        "measurable": float(scores[1]),
        "achievable": float(scores[2]),
        "time_bound": float(scores[3])
    }

    goal_scores = get_goal_scores(title)
    if not goal_scores:
        return {
            "title": title,
            "text": text,
            "criteria_scores": {},
            "average_score": 0
        }

    selected_scores = {k: v for k, v in criteria.items() if goal_scores[k] != 0}
    average_score = float(sum(selected_scores.values()) / len(selected_scores)) if selected_scores else 0

    result = {
        "title": title,
        "text": text,
        "criteria_scores": selected_scores,
        "average_score": average_score
    }
    return result

def normalize_weights(weights):
    total = sum(weights)
    return [w / total for w in weights]

def compute_similarity_scores(project, business, weights, threshold=0.20):
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

# API endpoint for SMART criteria evaluation
@app.route('/evaluate_smat_criteria', methods=['POST'])
def evaluate_smart_criteria_api():
    request_data = request.get_json()
    artifact_name = request_data.get('artifactName')
    elements = request_data.get('elements', {})

    evaluation_results = []
    total_average_score = 0
    count_of_scores = 0

    def evaluate_nested_elements(main_title, elements):
        for key, value in elements.items():
            full_key = f"{main_title} - {key}"
            if isinstance(value, str):
                evaluation_result = evaluate_smart_criteria(main_title, value)
                evaluation_result["nested_key"] = key
                if evaluation_result["average_score"] > 0:
                    nonlocal total_average_score, count_of_scores
                    total_average_score += evaluation_result["average_score"]
                    count_of_scores += 1
                evaluation_results.append(evaluation_result)
            elif isinstance(value, dict):
                evaluate_nested_elements(full_key, value)

    for key, value in elements.items():
        if isinstance(value, str):
            evaluation_result = evaluate_smart_criteria(key, value)
            if evaluation_result["average_score"] > 0:
                total_average_score += evaluation_result["average_score"]
                count_of_scores += 1
            evaluation_results.append(evaluation_result)
        elif isinstance(value, dict):
            evaluate_nested_elements(key, value)

    overall_average_score = total_average_score / count_of_scores if count_of_scores else 0

    return jsonify({
        "evaluation_results": evaluation_results,
        "overall_average_score": overall_average_score
    })

# API endpoint for project and business strategy analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    artifacts = data['artifacts']
    
    project_data = None
    business_data = None
    
    for artifact in artifacts:
        if artifact['artifactName'] == 'ProjectStrategy':
            project_data = artifact['GOCKS']
        elif artifact['artifactName'] == 'BusinessObjective':
            business_data = artifact['GOCKS']
    
    if project_data is None or business_data is None:
        return jsonify({"error": "Both 'ProjectStrategy' and 'BusinessObjective' artifacts must be provided."}), 400
    
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
    
    output = {
        "Total Feasibility Score": normalized_total_similarity_score,
        "artifacts": [
            {
                "artifactName": "ProjectStrategy",
                "GOCKS": {
                    key: {sub_key: {"Value": value, "Similarity Score": df_scores.loc[df_scores['Project Statement'] == f"{key} - {sub_key}", 'Weighted Average Similarity Score'].values[0]}
                          for sub_key, value in values.items()}
                    for key, values in project_data.items()
                }
            }
        ]
    }
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
