import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import evaluate
import numpy as np
import os
import pandas as pd
import fitz
import argparse
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',clean_up_tokenization_spaces=True)
accuracy_metric = evaluate.load("accuracy")

def chunk_text(text, tokenizer, max_length=512):
    # Encode without adding special tokens to get raw token ids
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunk_size = max_length - 2
    chunks = []
    len_text = 0
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        # Add special tokens back
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunk_text_str = tokenizer.decode(chunk_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # print(chunk_text_str)  
        chunks.append(chunk_text_str)
        len_text += len(chunk_text_str)
    return chunks

def expand_dataset(dataset):
    expanded_data = {"text": [], "label": []}
    for example in dataset:
        # Split the text into chunks
        chunks = chunk_text(example["text"], tokenizer, max_length=512)
        # For each chunk, store the chunk and the original label
        for chunk in chunks:
            expanded_data["text"].append(chunk)
            expanded_data["label"].append(example["label"])

    return Dataset.from_dict(expanded_data)

# Define a tokenization function for the chunks
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy["accuracy"]}

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            full_text += page.get_text().strip()

        return full_text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
def predict_label_for_pdf(pdf_path, model, tokenizer, device):
    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"No text extracted from {pdf_path}")
        return None

    # Split the full text into chunks that do not exceed 512 tokens
    chunks = chunk_text(text, tokenizer, max_length=512)
    # Store logits from each chunk
    all_logits = []
    for chunk in chunks:
        # Tokenize each chunk (padding and truncation ensure fixed size input)
        inputs = tokenizer(chunk, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  
            all_logits.append(logits)
    
    # Aggregate predictions by averaging the logits across chunks.
    aggregated_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0, keepdim=True)  
    predicted_label = torch.argmax(aggregated_logits, dim=1).item()
    return predicted_label, aggregated_logits

def process_pdfs_after_training(directory_path, output_csv, model, tokenizer, device):
    results = []
    
    # Iterate through all PDF files in the directory
    temp = {}
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            # print(f"Processing {filename}...")
            
            # Predict label for the PDF 
            predicted_label,_ty2 = predict_label_for_pdf(pdf_path, model, tokenizer, device)
            if predicted_label is None:
                continue  

            # Convert numeric prediction to string label
            label_map = {1: "YES", 0: "NO"}
            new_label = label_map.get(predicted_label, "UNKNOWN")
            results.append({"filename": filename, "predicted_label": new_label})
            
            temp[filename]=(_ty2)
    for i in temp:
        print(i, temp[i])
    # Save the results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
 
    # Load the model 
    domain_data_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    domain_data_model.load_state_dict(torch.load("models/domain_data_model.pth", map_location=device, weights_only=True), strict=False)
    domain_data_model = domain_data_model.to(device)

    # Evaluate the model on test data
    domain_data_model.eval()

    # Define the directory containing the test PDFs
    folder_test = "test"
    pdfs_dir = f"pdf_data/{folder_test}"

    # Create output directory if it doesn't exist
    if not os.path.exists("result"):
        os.makedirs("result")
    output_csv_path = f"result/predictions.csv"

    process_pdfs_after_training(
        directory_path=pdfs_dir,
        output_csv=output_csv_path,
        model=domain_data_model,       
        tokenizer=tokenizer,     
        device=device            
    )
