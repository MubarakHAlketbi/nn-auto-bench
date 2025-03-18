import json
import pandas as pd
import logging
import argparse
import os
import glob
from Levenshtein import distance as edit_distance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def calculate_field_metrics(annotation, preds, queried_labels):
    """
    Recalculate metrics for a single result by comparing predictions to annotations.
    
    Args:
        annotation (dict): Ground truth fields from the result.
        preds (dict): Predicted fields from the result.
        queried_labels (list): List of field names to evaluate.
    
    Returns:
        dict: Recalculated metrics (tp, fp, fn, file_accuracy, etc.).
    """
    metrics_dict = {}
    tp = 0.0
    fp = 0.0
    fn = 0.0
    total_gt_fields = len(annotation.get('fields', {}))
    total_pred_fields = len(preds) if isinstance(preds, dict) else 0

    annotation_fields = annotation.get('fields', {}) if isinstance(annotation, dict) else {}
    preds = preds if isinstance(preds, dict) else {}

    for key in queried_labels:
        gt_val = annotation_fields.get(key, {"value": ""}).get("value", "")
        gt_val = "" if gt_val is None else str(gt_val)

        pred_item = preds.get(key, {"value": ""})
        pred_val = pred_item.get("value", "") if isinstance(pred_item, dict) else str(pred_item)
        pred_val = "" if pred_val is None else str(pred_val)

        if gt_val or pred_val:
            edit_dist = edit_distance(pred_val, gt_val)
            max_len = max(len(pred_val), len(gt_val))
            score = 1 - (edit_dist / max_len if max_len > 0 else 1)
            tp += score
            fp += 1 - score
            fn += 1 - score
        elif gt_val == "" and pred_val == "":
            tp += 1.0

    file_accuracy = tp / len(queried_labels) if queried_labels else (1.0 if total_pred_fields == 0 and total_gt_fields == 0 else 0.0)

    metrics_dict = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "file_accuracy": file_accuracy,
        "total_gt_fields": total_gt_fields,
        "total_pred_fields": total_pred_fields
    }
    return metrics_dict

def process_single_file(input_file, output_dir, expected_entries):
    """Process a single JSONL file and save only the summary."""
    logger.info(f"Processing file: {input_file}")
    results = []
    
    # Read the input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line in {input_file}: {e}")
    except Exception as e:
        logger.error(f"Error reading file '{input_file}': {e}")
        return

    if not results:
        logger.error(f"No valid results found in {input_file}.")
        return

    # Convert to DataFrame for processing
    df_results = pd.DataFrame(results)

    # Recalculate metrics for each entry
    for index, row in df_results.iterrows():
        annotation = row.get('annotation', {'fields': {}})
        preds = row.get('pred', {})
        queried_labels = row.get('queried_labels', [])

        metrics = calculate_field_metrics(annotation, preds, queried_labels)
        for key, value in metrics.items():
            df_results.at[index, key] = value
        df_results.at[index, 'parsing_accuracy'] = 1 if isinstance(preds, dict) and preds else 0

    # Calculate aggregate metrics
    avg_accuracy = df_results['file_accuracy'].mean()
    parsing_accuracy = df_results['parsing_accuracy'].mean()
    correct_approved = df_results.get('correct_approved', pd.Series(0)).sum()
    incorrect_approved = df_results.get('incorrect_approved', pd.Series(0)).sum()
    total_fields = df_results.get('total_fields', pd.Series([len(row.get('queried_labels', [])) for _, row in df_results.iterrows()])).sum()

    # Use the file name (without .jsonl) as the model name
    base_name = os.path.basename(input_file)
    model_name = base_name.replace('.jsonl', '')

    # Prepare summary dictionary
    summary = {
        'model_name': model_name,
        'approval_accuracy': correct_approved / (correct_approved + incorrect_approved) if (correct_approved + incorrect_approved) > 0 else float('nan'),
        'approval_rate': (correct_approved + incorrect_approved) / total_fields if total_fields > 0 else 0.0,
        'parsing_accuracy': parsing_accuracy,
        'average_accuracy': avg_accuracy,
        'total_processed_entries': len(df_results),
        'missing_entries': expected_entries - len(df_results),
        'entries_with_parsing_errors': len(df_results[df_results['parsing_accuracy'] == 0])
    }

    # Save only the summary to JSON
    summary_file = os.path.join(output_dir, f"summary_{base_name.replace('.jsonl', '.json')}")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving summary to '{summary_file}': {e}")
        return

    # Log summary
    logger.info(f"Results for model: {model_name}")
    logger.info(f"Approval Accuracy: {summary['approval_accuracy']:.4f}, Approval Rate: {summary['approval_rate']:.4f}")
    logger.info(f"Parsing Accuracy: {summary['parsing_accuracy']:.4f}")
    logger.info(f"Average Accuracy: {summary['average_accuracy']:.4f}")
    logger.info(f"Total processed entries: {summary['total_processed_entries']}")
    if summary['missing_entries'] > 0:
        logger.warning(f"There are {summary['missing_entries']} missing entries in {input_file}.")
    if summary['entries_with_parsing_errors'] > 0:
        logger.warning(f"There are {summary['entries_with_parsing_errors']} entries with parsing errors in {input_file}.")

def parse_results(path, expected_entries=1000):
    """Parse results from a single file or all JSONL files in a directory."""
    output_dir = "parsed_results"
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}")
        return

    if os.path.isfile(path):
        # Process single file
        process_single_file(path, output_dir, expected_entries)
    elif os.path.isdir(path):
        # Process all JSONL files in directory
        jsonl_files = glob.glob(os.path.join(path, '*.jsonl'))
        if not jsonl_files:
            logger.warning(f"No JSONL files found in directory '{path}'.")
        for jsonl_file in jsonl_files:
            process_single_file(jsonl_file, output_dir, expected_entries)
    else:
        logger.error(f"Provided path '{path}' is neither a file nor a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse benchmark results from a JSONL file or directory and compute metrics.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a JSONL file or directory containing JSONL files"
    )
    parser.add_argument(
        "--expected_entries",
        type=int,
        default=1000,
        help="Expected number of entries in each result file (default: 1000)"
    )

    args = parser.parse_args()
    logger.info(f"Starting result parsing with arguments: {args}")

    parse_results(args.path, args.expected_entries)

    logger.info("Result parsing completed.")