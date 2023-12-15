
import pandas as pd

psgs_w100_file = "./data/open_domain_data/psgs_w100.tsv"

# output file
processed_file = "./data/open_domain_data/processed_psgs_w100.tsv"

# Read the "psgs_w100.tsv" file using pandas
df = pd.read_csv(psgs_w100_file, sep='\t', header=None, names=["id", "text", "title"],nrows=100000)

# Extract the text column
documents = df["text"]

# Set your desired chunk size and chunk overlap values
#
chunk_size = 1024
chunk_overlap = 20

# Function to preprocess and write data to a new file
def preprocess_and_write_data(documents, chunk_size, chunk_overlap, output_file):
    with open(output_file, "w") as outfile:
        for text in documents:
            # Process the text (e.g., tokenization, cleaning, etc.)
            # You can add your preprocessing steps here
            
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-chunk_overlap)]

            for chunk in chunks:
                outfile.write(chunk + '\n')

# Preprocess and write the data to the output file
preprocess_and_write_data(documents, chunk_size, chunk_overlap, processed_file)

print(f"Processed data written to {processed_file} with chunk_size {chunk_size} and overlap {chunk_overlap}")