

# Helper Methods
import re
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import json
from sklearn.calibration import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import Prompts as prompts
import numpy as np
import shutil


def create_balanced_dataset(df_reviews, num_reviews, random_seed=1):

    # Calculate the number of reviews per class
    num_reviews_per_class = num_reviews // 4
    print(f"number of reviews per class: {num_reviews_per_class}")
        
    # Filter and sample the specified number of reviews per class
    Bug = df_reviews[df_reviews['llm_annotation'] == "Bug"].sample(n=num_reviews_per_class, random_state=random_seed)
    Feature = df_reviews[df_reviews['llm_annotation'] == "Feature"].sample(n=num_reviews_per_class, random_state=random_seed)
    UserExperience = df_reviews[df_reviews['llm_annotation'] == "UserExperience"].sample(n=num_reviews_per_class, random_state=random_seed)
    Rating = df_reviews[df_reviews['llm_annotation'] == "Rating"].sample(n=num_reviews_per_class, random_state=random_seed)

    # Ensure we have enough reviews in each category
    if len(Bug) < num_reviews_per_class or len(Feature) < num_reviews_per_class or len(UserExperience) < num_reviews_per_class or len(Rating) < num_reviews_per_class:
        raise ValueError("Not enough reviews to meet the requested number per class")
    
    # Concatenate the samples from each class
    balanced_df = pd.concat([Bug, Feature, UserExperience, Rating]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return balanced_df

def read_json_file(file_path):
    try:
        with open(file_path, 'r',encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return []

# Function to write JSON data to a file
def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to append a JSON object to the file
def append_to_json(file_path, new_data):

    # Read the existing data from the file
    data = read_json_file(file_path)

    # Append the new JSON object to the list
    for review in new_data:
        data.append(review)

    # Write the updated data back to the file
    write_json_file(file_path, data)

def read_reviewes(input_file_path):
    reviews = []
    with open(input_file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        for review in data:
            reviews.append(review)
    return reviews

def extract_first_json(s):

    matches = re.findall(r'\{[^}]*\}', s, re.DOTALL)

    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        print("No valid JSON object found.")
    else:
        print("No match found.")

def process_reviews_classification(debug,input_file_path, output_file_path, pipeline,is_Explaining,max_length_value_inference,temperature_value,top_p_value,tokenizer):

    reviews = read_json_file(input_file_path)
    annotated_reviews = read_json_file(output_file_path)

    # Extract review IDs from annotated_reviews
    annotated_review_ids = [str(review["reviewId"]) + str(review["appId"]) for review in annotated_reviews]
    annotation_pending_reviews = []

    # Check if review is already annotated
    for review in reviews:
        combined_id = str(review["reviewId"]) + str(review["appId"])
        if combined_id not in annotated_review_ids:
            annotation_pending_reviews.append(review)

    print(f"Number of reviews: {len(reviews)}")
    print(f"Number of annotated reviews: {len(annotated_reviews)}")
    print(f"Number of annotation pending reviews: {len(annotation_pending_reviews)}")
    print(f"Annotated and annotation pending Sum: {len(annotated_reviews) + len(annotation_pending_reviews)}")

    processed_reviews = []

    # Process annotation pending reviews
    save_count = 0
    for review in tqdm(annotation_pending_reviews, desc="Processing Reviews"):

        # Get encoded prompt
        prompt=""
        if is_Explaining:
            prompt = prompts.generate_prompt_validation_with_Explanation(review)
        else:
            prompt = prompts.generate_prompt_validation(review)

        # Send prompt to LLM
        try:
            if debug:print(f"Prompt : {prompt}")
            response=None
            if tokenizer is not None:
                response = pipeline(prompt, do_sample=True ,return_full_text=False, num_return_sequences=1,max_new_tokens=max_length_value_inference,temperature=temperature_value,top_p=top_p_value,eos_token_id=tokenizer.eos_token_id)
            else:
                response = pipeline(prompt, do_sample=True ,return_full_text=False, num_return_sequences=1,max_new_tokens=max_length_value_inference,temperature=temperature_value,top_p=top_p_value)

            generated_text = response[0]["generated_text"]
            if debug:print(f"LLM Text: {generated_text}")
            # Extract the JSON object from the response
            data_dict = extract_first_json(generated_text)

            if is_Explaining:
                review["Explanation"] = data_dict["Explanation"]
            
            review["Prediction"] = data_dict["Class"]

            #Append the review to the processed reviews list is it is one of the classes
            if str(data_dict["Class"])=="Bug" or str(data_dict["Class"])=="Feature" or str(data_dict["Class"])=="UserExperience" or str(data_dict["Class"])=="Rating":
                processed_reviews.append(review)
            save_count += 1
            if save_count % 3 == 0:
                append_to_json(output_file_path, processed_reviews)
                processed_reviews = []
        except Exception as e:
            print(f"Error: {e}")
            append_to_json(output_file_path, processed_reviews)
            processed_reviews = []
            continue
    
    # Append the processed reviews to the output file
    append_to_json(output_file_path, processed_reviews)
    print("Done")

def process_and_save(debug,iteration,model_name, temperature, top_p, input_file_path, output_dir,pipeline,is_Explaining,max_length_value_inference,tokenizer=None):
    output_file_name = f"{model_name}_Temp_{temperature}_Top_{top_p}_Iteration_{iteration}.json"
    output_file_path = os.path.join(output_dir, output_file_name)

    continue_processing = True
    while continue_processing:
        process_reviews_classification(debug,input_file_path, output_file_path, pipeline, is_Explaining,max_length_value_inference,temperature,top_p,tokenizer)
        reviews = read_json_file(input_file_path)
        annotated_reviews = read_json_file(output_file_path)
        if len(reviews) == len(annotated_reviews):
            continue_processing = False
            print(f"Iteration {iteration} processing completed successfully.")


def process_files(model_name,output_dir, num_iterations, temperature, top_p):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    f1_micro_scores = []
    for iteration in range(1, num_iterations + 1):
        output_file_name = f"{model_name}_Temp_{temperature}_Top_{top_p}_Iteration_{iteration}.json"
        output_file_path = os.path.join(output_dir, output_file_name)
   
        with open(output_file_path, 'r') as file:
            data = json.load(file)
        
        print(f"\n=== Processing Iteration {iteration} ===\n")
        
        # Evaluate the results
        true_labels = [entry["reviewAnnotatorLabel"] for entry in data]
        predicted_labels = [entry["Prediction"] for entry in data]

        label_encoder = LabelEncoder()
        true_labels_encoded = label_encoder.fit_transform(true_labels)
        predicted_labels_encoded = label_encoder.transform(predicted_labels)

        # Calculate F1 score
        f1 = f1_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
        precision = precision_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
        recall = recall_score(true_labels_encoded, predicted_labels_encoded, average='weighted')
        f1_micro = f1_score(true_labels_encoded, predicted_labels_encoded, average='micro')
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_micro_scores.append(f1_micro)

        print("F1 Score:", f1)
        print("Precision Score:", precision)
        print("Recall Score:", recall)
        print("F1 Micro Score:", f1_micro)

        # Calculate confusion matrix
        confusion_mat = confusion_matrix(true_labels_encoded, predicted_labels_encoded)

        # Plot the confusion matrix as a heatmap
        labels = list(label_encoder.classes_)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Iteration {iteration}')
        plt.show()
    
    # Calculate the average F1 score
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1_micro = sum(f1_micro_scores) / len(f1_micro_scores)
    print("\n=== Average Scores ===")
    print("Average F1 Score:", avg_f1)
    print("Average Precision Score:", avg_precision)
    print("Average Recall Score:", avg_recall)
    print("Average F1 Micro Score:", avg_f1_micro)


def process_files_accuracy_measures(model_name,output_dir, num_iterations, temperature, top_p):
    class_wise_metrics = {}
    macro_avg_scores = {
        "precision": [],
        "recall": [],
        "f1_score": []
    }
    weighted_avg_scores = {
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    for iteration in range(1, num_iterations + 1):
        output_file_name = f"{model_name}_Temp_{temperature}_Top_{top_p}_Iteration_{iteration}.json"
        output_file_path = os.path.join(output_dir, output_file_name)
        
        # Load the JSON data from the file
        with open(output_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Evaluate the results
        true_labels = [entry["reviewAnnotatorLabel"] for entry in data]
        predicted_labels = [entry["Prediction"] for entry in data]

        label_encoder = LabelEncoder()
        true_labels_encoded = label_encoder.fit_transform(true_labels)
        predicted_labels_encoded = label_encoder.transform(predicted_labels)

        # Calculate overall evaluation metrics
        labels = list(label_encoder.classes_)

        # Calculate class-wise metrics
        class_report = classification_report(true_labels_encoded, predicted_labels_encoded, target_names=labels, output_dict=True)
        for label in labels:
            if label not in class_wise_metrics:
                class_wise_metrics[label] = {
                    "precision": [],
                    "recall": [],
                    "f1_score": []
                }
            class_wise_metrics[label]["precision"].append(class_report[label].get("precision", 0.0))
            class_wise_metrics[label]["recall"].append(class_report[label].get("recall", 0.0))
            class_wise_metrics[label]["f1_score"].append(class_report[label].get("f1-score", 0.0))

        # Collect macro average scores
        macro_avg_scores["precision"].append(class_report["macro avg"].get("precision", 0.0))
        macro_avg_scores["recall"].append(class_report["macro avg"].get("recall", 0.0))
        macro_avg_scores["f1_score"].append(class_report["macro avg"].get("f1-score", 0.0))

        # Collect weighted average scores
        weighted_avg_scores["precision"].append(class_report["weighted avg"].get("precision", 0.0))
        weighted_avg_scores["recall"].append(class_report["weighted avg"].get("recall", 0.0))
        weighted_avg_scores["f1_score"].append(class_report["weighted avg"].get("f1-score", 0.0))

    # Calculate the average for each class-wise metric
    average_class_wise_metrics = {
        label: {
            "precision": np.mean(metrics["precision"]),
            "recall": np.mean(metrics["recall"]),
            "f1_score": np.mean(metrics["f1_score"])
        }
        for label, metrics in class_wise_metrics.items()
    }

    # Calculate the overall macro and weighted average scores
    final_macro_avg_scores = {key: np.mean(val) for key, val in macro_avg_scores.items()}

    #print precision, recall, f1 score for each class and macro and weighted average scores as comma separated values in a single line given order of classes bug, feature, user experience, rating and macro avg
    print("Bug, Feature, User Experience, Rating, macro avg")
    comma_sep_values = ""
    for label in labels:
        comma_sep_values += f"{average_class_wise_metrics[label]['precision']}, {average_class_wise_metrics[label]['recall']}, {average_class_wise_metrics[label]['f1_score']}, "
    comma_sep_values += f"{final_macro_avg_scores['precision']}, {final_macro_avg_scores['recall']}, {final_macro_avg_scores['f1_score']}"
    print(comma_sep_values)

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')