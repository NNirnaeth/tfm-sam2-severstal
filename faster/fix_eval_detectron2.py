#!/usr/bin/env python3
"""
Fix the eval_detectron2.py file by removing the incorrect division by 100
"""

def fix_eval_file():
    # Read the file
    with open('eval_detectron2.py', 'r') as f:
        content = f.read()
    
    # Fix the metrics calculation
    old_code = '''        # Convert from percentage to decimal for consistency
        metrics = {
            'AP': bbox_results.get('AP', 0.0) / 100.0,
            'AP50': bbox_results.get('AP50', 0.0) / 100.0,
            'AP75': bbox_results.get('AP75', 0.0) / 100.0,
            'APs': bbox_results.get('APs', 0.0) / 100.0,
            'APm': bbox_results.get('APm', 0.0) / 100.0,
            'APl': bbox_results.get('APl', 0.0) / 100.0,
            'AR1': bbox_results.get('AR1', 0.0) / 100.0,
            'AR10': bbox_results.get('AR10', 0.0) / 100.0,
            'AR100': bbox_results.get('AR100', 0.0) / 100.0,
            'ARs': bbox_results.get('ARs', 0.0) / 100.0,
            'ARm': bbox_results.get('ARm', 0.0) / 100.0,
            'ARl': bbox_results.get('ARl', 0.0) / 100.0,
        }'''
    
    new_code = '''        # Detectron2 COCOEvaluator returns values in 0-1 scale, NOT percentages
        metrics = {
            'AP': bbox_results.get('AP', 0.0),
            'AP50': bbox_results.get('AP50', 0.0),
            'AP75': bbox_results.get('AP75', 0.0),
            'APs': bbox_results.get('APs', 0.0),
            'APm': bbox_results.get('APm', 0.0),
            'APl': bbox_results.get('APl', 0.0),
            'AR1': bbox_results.get('AR1', 0.0),
            'AR10': bbox_results.get('AR10', 0.0),
            'AR100': bbox_results.get('AR100', 0.0),
            'ARs': bbox_results.get('ARs', 0.0),
            'ARm': bbox_results.get('ARm', 0.0),
            'ARl': bbox_results.get('ARl', 0.0),
        }'''
    
    # Replace the code
    content = content.replace(old_code, new_code)
    
    # Write the fixed file
    with open('eval_detectron2.py', 'w') as f:
        f.write(content)
    
    print("Fixed eval_detectron2.py - removed incorrect division by 100")

if __name__ == "__main__":
    fix_eval_file()
