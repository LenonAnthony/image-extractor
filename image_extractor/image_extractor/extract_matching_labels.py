#!/usr/bin/env python3
import os
import json

def read_labels_file(labels_path):
    """
    Read the labels.txt file and return a dictionary mapping image numbers to their labels
    """
    labels_dict = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_name = parts[0]
                label = parts[1]
                # Extract number from image name (e.g., "1.png" -> 1)
                image_number = int(image_name.split('.')[0])
                labels_dict[image_number] = label
    return labels_dict

def main():
    # Paths
    base_path = os.path.join('math_images')  # Simplified path to math_images
    labels_path = os.path.join(base_path, 'labels.txt')
    json_path = 'math_images_list.json'
    output_path = 'matching_labels.txt'

    # Read the JSON file with image numbers
    with open(json_path, 'r') as f:
        image_numbers = json.load(f)

    # Read the labels file
    labels_dict = read_labels_file(labels_path)

    # Get all unique image numbers from all categories
    all_image_numbers = set()
    for category_numbers in image_numbers.values():
        all_image_numbers.update(category_numbers)

    # Write matching labels to new file
    with open(output_path, 'w', encoding='utf-8') as f:
        for image_number in sorted(all_image_numbers):
            if image_number in labels_dict:
                f.write(f"{image_number}.png\t{labels_dict[image_number]}\n")

    print(f"Extracted matching labels saved to {output_path}")
    print(f"Total matching entries: {len(all_image_numbers)}")

if __name__ == "__main__":
    main() 