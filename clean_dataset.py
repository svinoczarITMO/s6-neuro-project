import pandas as pd
import os

def clean_dataset(input_tsv, output_tsv):
    # Read the TSV file
    df = pd.read_csv(input_tsv, sep='\t')
    
    # Get the total number of entries before cleaning
    total_entries = len(df)
    
    # Check which files exist
    valid_entries = []
    base_dir = os.path.dirname(input_tsv)  # Get the directory containing the TSV file
    
    for index, row in df.iterrows():
        audio_path = os.path.join(base_dir, row['audio_path'])
        if os.path.exists(audio_path):
            valid_entries.append(index)
    
    # Create new dataframe with only valid entries
    cleaned_df = df.loc[valid_entries]
    
    # Save the cleaned dataset
    cleaned_df.to_csv(output_tsv, sep='\t', index=False)
    
    # Print statistics
    print(f"\nProcessing {os.path.basename(input_tsv)}:")
    print(f"Original dataset size: {total_entries}")
    print(f"Cleaned dataset size: {len(cleaned_df)}")
    print(f"Removed entries: {total_entries - len(cleaned_df)}")

if __name__ == "__main__":
    # Clean training dataset
    train_input = "podcast/podcast_train/raw_podcast_train.tsv"
    train_output = "podcast/podcast_train/cleaned_podcast_train.tsv"
    clean_dataset(train_input, train_output)
    
    # Clean test dataset
    test_input = "podcast/podcast_test/raw_podcast_test.tsv"
    test_output = "podcast/podcast_test/cleaned_podcast_test.tsv"
    clean_dataset(test_input, test_output) 