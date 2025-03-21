import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict
from tqdm import tqdm

def download_yelp_dataset(output_dir: str = 'data', max_reviews: int = 10000):
    """Download and prepare the Yelp dataset."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting to process Yelp dataset...")
    print(f"Will process {max_reviews} reviews for faster testing")
    
    # Process reviews with progress bar
    print("Processing reviews...")
    reviews = []
    with open(os.path.join(output_dir, 'yelp_academic_dataset_review.json'), 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Reading reviews")):
            if i >= max_reviews:  # Only process max_reviews
                break
            review = json.loads(line)
            reviews.append({
                'text': review['text'],
                'stars': review['stars'],
                'categories': review.get('categories', [])
            })
    
    print(f"Converting {len(reviews)} reviews to DataFrame...")
    # Convert to DataFrame
    df = pd.DataFrame(reviews)
    
    # Create conversation pairs with progress bar
    print("Creating conversation pairs...")
    conversations = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating conversations"):
        # Create a conversation pair
        conversation = {
            'input': f"User: What restaurants do you recommend for {', '.join(row['categories'])}?",
            'output': f"Assistant: Based on the reviews, I would recommend this place. {row['text']}"
        }
        conversations.append(conversation)
    
    print("Splitting into train and validation sets...")
    # Split into train and validation sets
    train_data, val_data = train_test_split(
        conversations,
        test_size=0.1,
        random_state=42
    )
    
    print("Saving processed data...")
    # Save processed data with progress bars
    print("Saving training data...")
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        for conv in tqdm(train_data, desc="Saving train data"):
            f.write(json.dumps(conv) + '\n')
    
    print("Saving validation data...")
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        for conv in tqdm(val_data, desc="Saving val data"):
            f.write(json.dumps(conv) + '\n')
    
    print("\nProcessing complete!")
    print(f"Processed {len(conversations)} conversations")
    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print("\nFiles saved in:", output_dir)

if __name__ == '__main__':
    download_yelp_dataset() 