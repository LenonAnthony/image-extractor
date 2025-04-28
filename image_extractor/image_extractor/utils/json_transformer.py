"""
Utility module for transforming JSON formats from the answer sheet extraction.
"""
import json
from pathlib import Path
from typing import Dict, Any


def transform_answer_sheet_json(input_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the answer sheet JSON from the original format:
    {
      "total_questions": 20,
      "questions": [
        {
          "question_number": 1,
          "selected_option": "C",
          "confidence": 0.98,
          "bounding_box": {
            "x1": 0.032,
            "y1": 0.023,
            "x2": 0.463,
            "y2": 0.076
          }
        },
        ...
      ],
      "elapsed": 21.57617473602295
    }

    To the new format:
    {
      "detections": {
        "1": {
          "name": "C",
          "bounding_box": {
            "x1": 0.032,
            "y1": 0.023,
            "x2": 0.463,
            "y2": 0.076
          },
          "confidence": 0.98
        },
        ...
      }
    }
    """
    result = {"detections": {}}
    
    for question in input_json.get("questions", []):
        question_number = str(question.get("question_number"))
        selected_option = question.get("selected_option")
        confidence = question.get("confidence")
        bounding_box = question.get("bounding_box", {})
        
        result["detections"][question_number] = {
            "name": selected_option,
            "bounding_box": bounding_box,
            "confidence": confidence
        }
    
    return result


def transform_file(input_path: str, output_path: str = None) -> None:
    """
    Transform a single JSON file from the original format to the new format.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the output JSON file. If None, will use the same filename
                     with "_transformed" appended before the extension.
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")
    
    if output_path is None:
        output_file = input_file.with_stem(f"{input_file.stem}")
    else:
        output_file = Path(output_path)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    output_data = transform_answer_sheet_json(input_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Transformed file saved to {output_file}")


def transform_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Transform all JSON files in a directory from the original format to the new format.
    
    Args:
        input_dir: Path to the directory containing input JSON files
        output_dir: Path to save the output JSON files. If None, will use the same directory
    """
    input_path = Path(input_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"Input directory {input_dir} not found or is not a directory")
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    for json_file in json_files:
        output_file = output_path / f"{json_file.stem}.json"
        transform_file(json_file, output_file)
    
    print(f"Transformed {len(json_files)} files to {output_path}")
