import json
import pandas as pd
import logging
import argparse
from Levenshtein import distance as edit_distance

# Configure logging to match typical benchmark output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_field_metrics(annotation, preds, queried_labels):
    """
    Recalculate metrics by comparing predictions to annotations.
    
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
    tp_strict = 0.0
    fp_strict = 0.0
    fn_strict = 0.0
    total_gt_fields = len(annotation.get('fields', {}))
    total_pred_fields = len(preds) if isinstance(preds, dict) else 0

    # Ensure annotation and preds are dictionaries
    annotation_fields = annotation.get('fields', {}) if isinstance(annotation, dict) else {}
    preds = preds if isinstance(preds, dict) else {}

    for key in queried_labels:
        # Extract ground truth value
        gt_val = annotation_fields.get(key, {"value": ""}).get("value", "")
        gt_val = "" if gt_val is None else str(gt_val)

        # Extract predicted value
        pred_item = preds.get(key, {"value": ""})
        pred_val = pred_item.get("value", "") if isinstance(pred_item, dict) else str(pred_item)
        pred_val = "" if pred_val is None else str(pred_val)

        # Calculate strict metrics (exact match)
        if pred_val == gt_val:
            tp_strict += 1.0
        else:
            fp_strict += 1.0
            fn_strict += 1.0

        # Calculate lenient metrics (edit distance-based)
        if gt_val or pred_val:  # Only compute if at least one value exists
            edit_dist = edit_distance(pred_val, gt_val)
            max_len = max(len(pred_val), len(gt_val))
            score = 1 - (edit_dist / max_len if max_len > 0 else 1)
            tp += score
            fp += 1 - score
            fn += 1 - score
        elif gt_val == "" and pred_val == "":
            tp += 1.0  # Both empty, consider it a match
            tp_strict += 1.0

    # Calculate file accuracy
    if len(queried_labels) > 0:
        file_accuracy = tp / len(queried_labels)
    else:
        file_accuracy = 1.0 if total_pred_fields == 0 and total_gt_fields == 0 else 0.0

    metrics_dict = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "file_accuracy": file_accuracy,
        "tp_strict": tp_strict,
        "fp_strict": fp_strict,
        "fn_strict": fn_strict,
        "total_gt_fields": total_gt_fields,
        "total_pred_fields": total_pred_fields
    }
    return metrics_dict

def parse_results(result_file, expected_entries=1000):
    """
    Parse the result.jsonl file, recalculate metrics, and summarize results.
    
    Args:
        result_file (str): Path to the result.jsonl file.
        expected_entries (int): Expected number of entries (default: 1000).
    """
    # Step 1: Read the JSONL file
    results = []
    try:
        with open(result_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    except FileNotFoundError:
        logger.error(f"Result file '{result_file}' not found.")
        return
    except Exception as e:
        logger.error(f"Error reading result file '{result_file}': {e}")
        return

    # Convert to DataFrame for easier analysis
    df_results = pd.DataFrame(results)

    # Step 2: Recalculate metrics for each result
    for index, row in df_results.iterrows():
        annotation = row.get('annotation', {'fields': {}})
        preds = row.get('pred', {})
        queried_labels = row.get('queried_labels', [])
        
        # Recalculate metrics
        metrics = calculate_field_metrics(annotation, preds, queried_labels)
        
        # Update DataFrame with recalculated metrics
        for key, value in metrics.items():
            df_results.at[index, key] = value

        # Update parsing accuracy (assume 1 if parsing succeeded, 0 if it failed)
        df_results.at[index, 'parsing_accuracy'] = 1 if isinstance(preds, dict) else 0

    # Step 3: Calculate aggregate metrics
    avg_accuracy = df_results['file_accuracy'].mean()
    parsing_accuracy = df_results['parsing_accuracy'].mean()
    correct_approved = df_results.get('correct_approved', pd.Series(0)).sum()
    incorrect_approved = df_results.get('incorrect_approved', pd.Series(0)).sum()
    total_fields = df_results.get('total_fields', pd.Series(len(queried_labels))).sum()

    # Step 4: Check for missing entries and errors
    actual_entries = len(df_results)
    missing_entries = expected_entries - actual_entries
    errors = df_results[df_results['parsing_accuracy'] == 0]
    error_count = len(errors)

    # Step 5: Generate and log the summary
    logger.info(f"Results for model: mistral-ocr")
    if correct_approved + incorrect_approved == 0:
        approval_accuracy = float('nan')
        approval_rate = 0.0
    else:
        approval_accuracy = correct_approved / (correct_approved + incorrect_approved)
        approval_rate = (correct_approved + incorrect_approved) / total_fields

    logger.info(f"Approval Accuracy: {approval_accuracy:.4f}, Approval Rate: {approval_rate:.4f}")
    logger.info(f"Parsing Accuracy: {parsing_accuracy:.4f}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Total processed entries: {actual_entries}")
    logger.info(f"Missing entries: {missing_entries}")
    logger.info(f"Entries with parsing errors: {error_count}")

    # Report missing entries
    if missing_entries > 0:
        logger.warning(f"There are {missing_entries} missing entries. "
                       f"Expected {expected_entries} results, but only {actual_entries} were found.")

    # Report parsing errors
    if error_count > 0:
        logger.warning(f"There are {error_count} entries with parsing errors:")
        for index, row in errors.iterrows():
            logger.warning(f"Error in entry: {row.get('path', f'Index {index}')}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse and recalculate benchmark results from a JSONL file.")
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to the result JSONL file (e.g., results/benchmark_results_mistral-ocr.jsonl)"
    )
    parser.add_argument(
        "--expected_entries",
        type=int,
        default=1000,
        help="Expected number of entries in the result file (default: 1000)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the parsing function
    parse_results(args.result_file, args.expected_entries)