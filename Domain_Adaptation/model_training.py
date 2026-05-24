import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pandas as pd
import fitz
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',clean_up_tokenization_spaces=True)

def chunk_text(text, tokenizer, max_length=512):
    # Encode without adding special tokens to get raw token ids
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Reserve two tokens for [CLS] and [SEP]
    chunk_size = max_length - 2
    chunks = []
    len_text = 0
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i+chunk_size]
        # Add special tokens back
        chunk_tokens = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
        chunk_text_str = tokenizer.decode(chunk_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        # print(chunk_text_str)  # Debugging: print the chunked text
        chunks.append(chunk_text_str)
        len_text += len(chunk_text_str)
    # Remove any leading/trailing whitespace
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
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

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

    # Tokenize the text
    inputs = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    # Move tensors to the device (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Predict label
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label

def process_pdfs_after_training(directory_path, output_csv, model, tokenizer, device):
    results = []
    
    # Iterate through all PDF files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing {filename}...")
            
            # Predict label for the PDF 
            predicted_label = predict_label_for_pdf(pdf_path, model, tokenizer, device)
            if predicted_label is None:
                continue  
            
            # Convert numeric prediction to string label
            label_map = {1: "YES", 0: "NO"}
            new_label = label_map.get(predicted_label, "UNKNOWN")
            results.append({"filename": filename, "predicted_label": new_label})
    
    # Save the results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on a limited number of samples per class.")
    parser.add_argument("--count", type=int, default=10, help="Number of 'YES' samples to use")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--filename", type=str, default="csv_data/combined.csv", help="Filename for the dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    domain_data_path = f"{args.filename}"
    domain_data = pd.read_csv(domain_data_path).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Filter and limit samples by label
    yes_samples = domain_data[domain_data["label"] == 1].head(args.count//2)
    no_samples = domain_data[domain_data["label"] == 0].head(args.count//2)
    print(f"Training on {len(yes_samples)} YES and {len(no_samples)} NO samples (total {len(domain_data)}).")

    # Combine and shuffle
    domain_data = pd.concat([yes_samples, no_samples]).sample(frac=1).reset_index(drop=True)
    print(f"Total samples after filtering: {len(domain_data)}")

    # Convert your DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(domain_data)
    dataset_chunked = Dataset.from_dict(domain_data.to_dict(orient='list'))
    print(f"Total samples after chunking: {len(dataset_chunked)}")

    dataset_tokenized = dataset_chunked.map(tokenize_function, batched=True)
    dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset_tokenized = dataset_tokenized.shuffle(seed=42)
    
    # Load the model 
    domain_data_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    domain_data_model.load_state_dict(torch.load("../Transfer_Learning/models/first_model.pth", map_location=device, weights_only=True), strict=False)
    domain_data_model = domain_data_model.to(device)

    # Training arguments
    print("-------------------------------------")
    print("Training: ")
    training_args = TrainingArguments(
        output_dir="./temp",
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=2
    )

    # Initialize Trainer
    trainer = Trainer(
        model=domain_data_model,
        args=training_args,
        train_dataset=dataset_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    # Fine-tune the model on the chunked training data
    trainer.train()

    print("-------------------------------------")
    print("Evaluation: ")
    domain_data_model.eval()

    # Save the model
    model_save_path = f"models/domain_data_model.pth"
    torch.save(domain_data_model.state_dict(), model_save_path)
    
    print("DONE")