import os
import pandas as pd
import fitz  # PyMuPDF
import pandas as pd
    
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

def create_dataset(folder_names, output_csv = None):
    data = []
    original_folder = "../Transfer_Learning/pdf_data/"
    for i in range(len(folder_names)):
        yes_pdfs_dir = os.path.join(original_folder, folder_names[i], "YES")
        no_pdfs_dir = os.path.join(original_folder, folder_names[i], "NO")
        
        # Process yes PDFs (Yes can be medical, waste, etc.)
        for filename in os.listdir(yes_pdfs_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(yes_pdfs_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    data.append({"text": text, "label": 1})

        # Process no PDFs
        for filename in os.listdir(no_pdfs_dir):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(no_pdfs_dir, filename)
                text = extract_text_from_pdf(pdf_path)
                if text:
                    data.append({"text": text, "label": 0})

    print(f"Processed {len(data)} PDFs in total.")
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save the dataset to a CSV file
    # file_name = f"combined_{len(data)}.csv"
    file_name = f"combined.csv" 
    if output_csv is None:
        output_csv = os.path.join("../Transfer_Learning/csv_data", file_name)
    df.to_csv(output_csv, index=False)
    print(f"Dataset saved to {output_csv}")

if __name__ == "__main__":
    # Add the folder names, it can concatenee different dataset into one csv file
    file_names = ['train']
    create_dataset(file_names)

