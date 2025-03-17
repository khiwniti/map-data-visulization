import json
import os
import sys

def print_json_structure(obj, prefix="", max_depth=3, current_depth=0):
    """Print the structure of a JSON object with sample values"""
    if current_depth >= max_depth:
        return
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}: {type(value).__name__}")
                if isinstance(value, dict):
                    print_json_structure(value, prefix + "  ", max_depth, current_depth + 1)
                elif isinstance(value, list) and value:
                    print(f"{prefix}  [length: {len(value)}]")
                    if value and isinstance(value[0], (dict, list)):
                        print_json_structure(value[0], prefix + "    ", max_depth, current_depth + 1)
                    else:
                        sample = value[0] if value else None
                        print(f"{prefix}  Sample item: {type(sample).__name__} - {str(sample)[:50]}")
            else:
                sample = str(value)[:50] + "..." if value and len(str(value)) > 50 else str(value)
                print(f"{prefix}{key}: {type(value).__name__} - {sample}")
    elif isinstance(obj, list) and obj:
        print(f"{prefix}[length: {len(obj)}]")
        if obj and isinstance(obj[0], (dict, list)):
            print_json_structure(obj[0], prefix + "  ", max_depth, current_depth + 1)
        else:
            sample = obj[0] if obj else None
            print(f"{prefix}Sample item: {type(sample).__name__} - {str(sample)[:50]}")

def extract_sample_json(file_path, output_file=None):
    """Extract a small sample of the JSON structure for debugging"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a simplified version with just the structure
        simplified = {}
        
        def simplify(obj, max_items=3):
            if isinstance(obj, dict):
                result = {}
                for i, (key, value) in enumerate(obj.items()):
                    if i >= max_items:
                        result["..."] = f"{len(obj) - max_items} more keys"
                        break
                    result[key] = simplify(value, max_items)
                return result
            elif isinstance(obj, list):
                if not obj:
                    return []
                if len(obj) > max_items:
                    return [simplify(obj[0], max_items), f"... {len(obj) - 1} more items"]
                return [simplify(item, max_items) for item in obj[:max_items]]
            else:
                return f"{type(obj).__name__}: {str(obj)[:30]}" if obj else None
        
        simplified = simplify(data)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(simplified, f, indent=2)
            print(f"Saved simplified structure to {output_file}")
        
        return simplified
    
    except Exception as e:
        print(f"Error extracting sample: {str(e)}")
        return None

def find_restaurant_paths(data, max_depth=5):
    """Find all possible paths that might contain restaurant data"""
    paths = []
    
    def search_path(obj, current_path=None, depth=0):
        if current_path is None:
            current_path = []
        
        if depth >= max_depth:
            return
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = current_path + [key]
                
                # Check if this might be a restaurant object
                if isinstance(value, dict) and any(k in value for k in ['name', 'address', 'cuisine', 'rating']):
                    paths.append((new_path, 'single_restaurant'))
                
                # Check if this is a list of potential restaurants
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    if any(k in value[0] for k in ['name', 'address', 'cuisine', 'rating', 'id']):
                        paths.append((new_path, 'restaurant_list', len(value)))
                
                search_path(value, new_path, depth + 1)
        
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            # Check if this list contains restaurant-like objects
            if any(k in obj[0] for k in ['name', 'address', 'cuisine', 'rating', 'id']):
                paths.append((current_path, 'restaurant_list', len(obj)))
            
            # Continue searching in the first item
            search_path(obj[0], current_path + ['[0]'], depth + 1)
    
    search_path(data)
    return paths

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_json_structure.py <json_file>")
        return
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Analyzing structure of {file_path}:")
        print_json_structure(data)
        
        # Find potential restaurant paths
        print("\nSearching for potential restaurant data paths:")
        paths = find_restaurant_paths(data)
        
        if paths:
            print("\nPotential restaurant data found at:")
            for path_info in paths:
                if len(path_info) == 2:
                    path, type_info = path_info
                    print(f"  Path: {' -> '.join(path)}, Type: {type_info}")
                else:
                    path, type_info, count = path_info
                    print(f"  Path: {' -> '.join(path)}, Type: {type_info}, Count: {count}")
                
                # Extract a sample
                current = data
                for key in path:
                    if key == '[0]' and isinstance(current, list) and current:
                        current = current[0]
                    elif isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                
                if current:
                    if isinstance(current, list) and current:
                        sample = current[0] if current else None
                        print(f"  Sample keys: {list(sample.keys())[:5] if isinstance(sample, dict) else type(sample)}")
                    elif isinstance(current, dict):
                        print(f"  Sample keys: {list(current.keys())[:5]}")
        else:
            print("No potential restaurant data paths found.")
        
        # Save a simplified structure for debugging
        output_file = f"structure_{os.path.basename(file_path)}.json"
        extract_sample_json(file_path, output_file)
        
    except Exception as e:
        print(f"Error analyzing file: {str(e)}")

if __name__ == "__main__":
    main()