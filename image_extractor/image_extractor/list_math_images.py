#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import json

def get_image_numbers(directory):
    """
    Extracts numbers from PNG filenames in the specified directory.
    
    Args:
        directory (str): Path to the directory containing PNG files
    
    Returns:
        list: List of image numbers (as integers)
    """
    numbers = []
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return numbers
    
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            # Extract the number from the filename using regex
            match = re.search(r'(\d+)(?:\(\d+\))?.png$', filename)
            if match:
                numbers.append(int(match.group(1)))
    
    return sorted(numbers)

def main():
    # Base path to the math_images directory
    base_path = os.path.join(os.getcwd(), 'image-extractor', 'image_extractor', 'image_extractor', 'math_images')
    
    # Check if the path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Math images directory not found at {base_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Contents of current directory: {os.listdir(os.getcwd())}")
        return
        
    print(f"Using math_images directory: {base_path}")
    
    # Categories to process
    categories = ['horizontais', 'verticais', 'ruins']
    
    # Dictionary to store results
    results = defaultdict(list)
    
    # Process each category
    for category in categories:
        path = os.path.join(base_path, category)
        image_numbers = get_image_numbers(path)
        results[category] = image_numbers
        print(f"Found {len(image_numbers)} images in {category}")
    
    # Save results to a JSON file
    output_file = 'math_images_list.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Print some statistics
    total_images = sum(len(nums) for nums in results.values())
    print(f"\nTotal images across all categories: {total_images}")
    
    # Print a summary of the first few numbers in each category
    for category, numbers in results.items():
        preview = numbers[:5]
        print(f"\n{category.capitalize()}: {len(numbers)} images")
        print(f"Sample numbers: {preview}...")

if __name__ == "__main__":
    main()