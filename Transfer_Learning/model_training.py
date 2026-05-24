import pandas as pd
import numpy as np
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',clean_up_tokenization_spaces=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

def split_data(split_count, yes_data, no_data, seed=42):
    train_count_per_class = split_count // 2
    val_count_per_class = train_count_per_class // 3

    required_per_class = train_count_per_class + val_count_per_class
    if len(yes_data) < required_per_class or len(no_data) < required_per_class:
        raise ValueError(f"Not enough samples: need at least {required_per_class} for both YES and NO classes.")

    yes_train = yes_data.iloc[:train_count_per_class]
    no_train = no_data.iloc[:train_count_per_class]

    yes_val = yes_data.iloc[train_count_per_class:train_count_per_class + val_count_per_class]
    no_val = no_data.iloc[train_count_per_class:train_count_per_class + val_count_per_class]

    train_data = pd.concat([yes_train, no_train])
    val_data = pd.concat([yes_val, no_val])

    train_data["split"] = "train"
    val_data["split"] = "val"

    used_indices = set(train_data.index).union(val_data.index)
    all_data = pd.concat([yes_data, no_data])
    test_data = all_data[~all_data.index.isin(used_indices)].copy()
    test_data["split"] = "test"

    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"Train: {len(train_data)} (YES: {sum(train_data.label==1)}, NO: {sum(train_data.label==0)})")
    print(f"Val:   {len(val_data)} (YES: {sum(val_data.label==1)}, NO: {sum(val_data.label==0)})")
    print(f"Test:  {len(test_data)} (YES: {sum(test_data.label==1)}, NO: {sum(test_data.label==0)})")

    return train_data, val_data, test_data


def data_conversion(data, split_count=800, seed=42):
    # Separate YES and NO data
    yes_data = data[data["label"] == 1].reset_index(drop=True)
    no_data = data[data["label"] == 0].reset_index(drop=True)

    # Shuffle both classes before splitting
    yes_data = yes_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    no_data = no_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Use your custom split
    train_data, val_data, test_data = split_data(split_count, yes_data, no_data, seed)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_data.to_dict(orient='list'))
    val_dataset = Dataset.from_dict(val_data.to_dict(orient='list'))
    test_dataset = Dataset.from_dict(test_data.to_dict(orient='list'))

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4 )
    val_dataset = val_dataset.map(tokenize_function, batched=True, num_proc=4 )
    test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4 )

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset, val_dataset, test_dataset


def evaluated_test_result(predictions):
    # Extract logits and true labels
    logits = predictions.predictions
    predicted_labels = np.argmax(logits, axis=1)  # Predicted class
    true_labels = predictions.label_ids           # True labels

    cm = confusion_matrix(true_labels, predicted_labels)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
    print("----------------------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) 
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# first fine-tuning on the dataset, then we will use the model to predict the next dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on a limited number of samples per class.")
    parser.add_argument("--count", type=int, default=20, help="Number of 'YES' samples to use")
    parser.add_argument("--filename", type=str, default="csv_data/combined.csv", help="Path to the CSV file containing the data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    firstft_data_path = f"{args.filename}"
    firstft_data = pd.read_csv(firstft_data_path).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    unique_labels = list(set(firstft_data['label']))
    print("Total number of papers: ", len(firstft_data['text']))
    print("----------------------------------------")
    
    train_dataset, val_dataset, test_dataset = data_conversion(firstft_data, split_count=args.count, seed=args.seed)
    print("----------------------------------------")
    print("Start training:")
    # Initialize the model
    firstft_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(unique_labels))
    firstft_model = firstft_model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./temp",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=None,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy", 
        save_total_limit=2
    )

    trainer = Trainer(
        model=firstft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    print("------------------------------------")
    print("Evaluation：")
    firstft_model.eval()

    # Save the model
    # model_save_path = f"models/first_model.pth"
    model_save_path = f"models/first_model.pth"
    torch.save(firstft_model.state_dict(), model_save_path)

    # Predict on the test dataset
    predictions = trainer.predict(test_dataset)
    evaluated_test_result(predictions)
    

