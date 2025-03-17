import json
import os
import pandas as pd
from pathlib import Path

def process_restaurant_data(input_file, output_dir):
    """
    Process restaurant data from JSON file and save each restaurant as a separate Parquet file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_dir (str): Directory to save the Parquet files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract restaurant entities
    restaurants = data.get('page', {}).get('entities', [])
    
    print(f"Found {len(restaurants)} restaurants in the file")
    
    # Process each restaurant
    for restaurant in restaurants:
        # Get the publicId as the unique identifier
        public_id = restaurant.get('publicId')
        
        if not public_id:
            print(f"Skipping restaurant with no publicId: {restaurant.get('id')}")
            continue
        
        # Convert restaurant to DataFrame
        df = pd.DataFrame([restaurant])
        
        # Save as Parquet file
        output_file = os.path.join(output_dir, f"restaurant_{public_id}.parquet")
        df.to_parquet(output_file, index=False)
        
        print(f"Saved restaurant {public_id} to {output_file}")

def process_multiple_files(input_dir, output_dir, file_pattern="*.json"):
    """
    Process multiple JSON files in a directory.
    
    Args:
        input_dir (str): Directory containing JSON files
        output_dir (str): Directory to save the Parquet files
        file_pattern (str): Pattern to match JSON files
    """
    # Get all JSON files in the input directory
    json_files = list(Path(input_dir).glob(file_pattern))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    for json_file in json_files:
        print(f"Processing {json_file}...")
        process_restaurant_data(str(json_file), output_dir)

if __name__ == "__main__":
    # Example usage for a single file
    # process_restaurant_data("LMWN/1-1.json", "restaurants_parquet")
    
    # Example usage for multiple files
    process_multiple_files("LMWN", "restaurants_parquet")