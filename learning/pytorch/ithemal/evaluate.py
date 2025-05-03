#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Keep print function behavior for consistency if desired,
# but standard Python 2 print statements will also work below.
from __future__ import print_function

import argparse
import binascii
import common_libs.utilities as ut
import copy
import data.data_cost as dt
import ithemal_utils
import multiprocessing
import os
import subprocess
import sys
import threading
import torch
import torch.nn.functional as F
import warnings
import csv
import time
import Queue  # For exception handling if needed

# Assuming the tokenizer path is correctly set via environment variable
_TOKENIZER = os.path.join(
    os.environ.get("ITHEMAL_HOME", "."), "data_collection", "build", "bin", "tokenizer"
)

# --- Functions reused/adapted from predict.py ---


def load_model_and_data(model_file, model_data_file):
    """Loads the model architecture and trained weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        # Load model structure and associated data (like tokenizers)
        (model, data) = ithemal_utils.load_model_and_data(model_file)

    # Load the trained weights (state dictionary)
    # Assuming torch.load works similarly in Python 2 compatible versions
    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()

    # Filter state_dict to only include keys present in the current model_dict
    new_model_dict = {
        k: v
        for (k, v) in state_dict.get("model", state_dict).items()
        if k in model_dict
    }
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    # Set model to evaluation mode
    model.eval()

    return (model, data)


def datum_of_code(data, block_hex):
    """
    Tokenizes and preprocesses a hex block string into a DataItem the model can use.
    Returns None if tokenization fails.
    """
    try:
        # Use the external tokenizer binary
        xml = subprocess.check_output([_TOKENIZER, block_hex, "--token"])
        # We don't need the intel disassembly for prediction/loss calculation
        intel = ""  # Use a dummy value
    except subprocess.CalledProcessError as e:
        # Use Python 2 style formatting
        print(
            "Warning: Tokenizer failed for hex {}...: {}".format(block_hex[:20], e),
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            "Warning: Unexpected error during tokenization for hex {}...: {}".format(
                block_hex[:20], e
            ),
            file=sys.stderr,
        )
        return None

    # Create a temporary raw_data entry for prepare_data
    data.raw_data = [(-1, -1, intel, xml)]
    data.data = []  # Clear previous data items if any
    try:
        # Prepare the data item using the loaded token mappings
        data.prepare_data(fixed=True, progress=False)
        if not data.data:
            print(
                "Warning: prepare_data resulted in empty data for hex {}...".format(
                    block_hex[:20]
                ),
                file=sys.stderr,
            )
            return None
        return data.data[-1]  # Return the processed DataItem
    except ValueError as e:
        print(
            "Warning: Error preparing data (possibly UNK token) for hex {}...: {}".format(
                block_hex[:20], e
            ),
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            "Warning: Unexpected error preparing data for hex {}...: {}".format(
                block_hex[:20], e
            ),
            file=sys.stderr,
        )
        return None


# --- New functions for CSV processing and loss calculation ---

# ProcessResult Tuple definition is implicit in Python 2


def process_line_worker(model_file, model_data_file, input_queue, output_queue):
    """
    Worker process function. Loads model, reads lines from input_queue,
    processes them, and puts results into output_queue.
    """
    try:
        # Each worker loads its own copy of the model and data params
        model, data = load_model_and_data(model_file, model_data_file)
        # GPU handling would be similar if torch version supports it on Py2
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)

        while True:
            line_tuple = input_queue.get()
            if line_tuple is None:  # Sentinel value indicates end of input
                break

            line_num, line = line_tuple
            if not line:
                continue

            parts = line.strip().split(",")
            if len(parts) != 2:
                print(
                    "Warning: Skipping malformed line {}: {}".format(
                        line_num, line.strip()
                    ),
                    file=sys.stderr,
                )
                output_queue.put((None, None))  # Signal failure for this line
                continue

            block_hex, true_val_str = parts
            try:
                true_value = float(true_val_str)
            except ValueError:
                print(
                    "Warning: Skipping line {} with non-float true value: {}".format(
                        line_num, true_val_str
                    ),
                    file=sys.stderr,
                )
                output_queue.put((None, None))
                continue

            # Get the processed DataItem
            datum = datum_of_code(data, block_hex)

            if datum is None:
                # datum_of_code already printed a warning
                output_queue.put((None, None))  # Signal failure
                continue
            
            try:
                # Perform prediction (no gradients needed for evaluation)
                with torch.no_grad():
                    # prediction_tensor = model(datum.x.to(device)) # If using GPU
                    prediction_tensor = model(datum)

                prediction = prediction_tensor.item()  # Get scalar value
                output_queue.put((prediction, true_value))
                # Clean up references if the model requires it
                if hasattr(model, "remove_refs"):
                    model.remove_refs(datum)

            except Exception as e:
                print(
                    "Warning: Model prediction failed for line {} (hex {}...): {}".format(
                        line_num, block_hex[:20], e
                    ),
                    file=sys.stderr,
                )
                output_queue.put((None, None))  # Signal failure

    except Exception as e:
        print("FATAL: Worker process failed: {}".format(e), file=sys.stderr)
        # Ensure the output queue gets a sentinel if this worker dies unexpectedly
        output_queue.put(None)


def collect_results(output_queue, total_lines):
    """Collects results from the output queue."""
    results = []
    processed_count = 0
    while processed_count < total_lines:
        result = output_queue.get()
        if result is None:  # Check if a worker sent a sentinel early (due to error)
            print(
                "Warning: Received early termination signal from a worker.",
                file=sys.stderr,
            )
            pass
        else:
            results.append(result)
        processed_count += 1
        # Use \r for progress indicator, works in most terminals
        print("\rProcessed: {}/{}".format(processed_count, total_lines), end="")
        sys.stdout.flush()  # Ensure it prints immediately
    print()  # Newline after progress indicator
    return results


def calculate_loss(results):
    """Calculates Mean Absolute Percentage Error (MAPE) from collected results."""
    predictions = []
    true_values = []
    valid_count = 0
    fail_count = 0
    zero_true_count = 0

    for pred, true in results:
        if pred is not None and true is not None:
            # Check if the true value is zero or very close to zero
            if abs(true) < 1e-9:  # Use a small epsilon
                zero_true_count += 1
                valid_count += 1  # Still counts as valid, but excluded from MAPE
            else:
                predictions.append(pred)
                true_values.append(true)
                valid_count += 1
        else:
            fail_count += 1

    if not predictions:  # Check if we have any valid non-zero true values
        print(
            "Error: No valid predictions with non-zero true values were made.",
            file=sys.stderr,
        )
        return None, valid_count, fail_count, zero_true_count

    # Convert lists to PyTorch tensors
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    true_values_tensor = torch.tensor(true_values, dtype=torch.float32)

    # Calculate Absolute Percentage Error: |predicted - actual| / |actual|
    absolute_percentage_error = torch.abs(
        (predictions_tensor - true_values_tensor) / true_values_tensor
    )

    # Calculate the mean
    mape_loss = torch.mean(absolute_percentage_error) * 100  # Express as percentage

    return mape_loss.item(), valid_count, fail_count, zero_true_count


# --- Main execution logic ---


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Ithemal model performance against a CSV dataset."
    )
    parser.add_argument(
        "--model", help="Path to the model architecture file (.dump)", required=True
    )
    parser.add_argument(
        "--model-data",
        help="Path to the trained model weights file (.mdl)",
        required=True,
    )
    parser.add_argument(
        "--input-file",
        help="Path to the input CSV file (format: hex_code,true_value)",
        required=True,
    )
    parser.add_argument(
        "--parallel",
        help="Number of parallel worker processes",
        type=int,
        default=multiprocessing.cpu_count(),
    )
    parser.add_argument(
        "--skip-header",
        help="Skip the first line of the CSV file",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # File existence checks
    if not os.path.exists(args.input_file):
        print(
            "Error: Input file not found: {}".format(args.input_file), file=sys.stderr
        )
        sys.exit(1)
    if not os.path.exists(args.model):
        print("Error: Model file not found: {}".format(args.model), file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model_data):
        print(
            "Error: Model data file not found: {}".format(args.model_data),
            file=sys.stderr,
        )
        sys.exit(1)
    if not os.path.exists(_TOKENIZER):
        print(
            "Error: Tokenizer executable not found at {}".format(_TOKENIZER),
            file=sys.stderr,
        )
        print(
            "Ensure ITHEMAL_HOME is set correctly and the tokenizer is built.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Setup multiprocessing queues
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    # --- Start worker processes ---
    print("Starting {} worker processes...".format(args.parallel))
    workers = []
    for _ in range(args.parallel):
        p = multiprocessing.Process(
            target=process_line_worker,
            args=(args.model, args.model_data, input_queue, output_queue),
        )
        p.daemon = True  # Allow main process to exit even if workers hang
        p.start()
        workers.append(p)

    # --- Read input file and feed the queue ---
    print("Reading input file: {}".format(args.input_file))
    line_count = 0
    total_lines = 0
    try:
        with open(args.input_file, "r") as infile:
            # Count lines first
            if args.skip_header:
                try:
                    next(infile)  # Skip header for counting
                    line_count = -1
                except StopIteration:
                    pass  # Empty file after header
            initial_pos = infile.tell()
            total_lines = sum(1 for _ in infile) + line_count
            infile.seek(initial_pos)  # Reset file pointer
            print("Found {} lines to process.".format(total_lines))

            if args.skip_header:  # Re-skip header if necessary and file wasn't empty
                if total_lines >= 0:  # Check if there are lines beyond potential header
                    try:
                        next(infile)
                    except StopIteration:
                        pass  # Handle case where file only had a header
            
            # Feed the queue
            for i, line in enumerate(infile):
                input_queue.put((i + 1, line))  # Send line number for logging
                line_count += 1
                print("\rReading input: Line %d / %d (%.1f%%)" % (i, total_lines, (float(i) / total_lines) * 100), end="")
                sys.stdout.flush()

    except IOError as e:  # Use IOError for file errors in Python 2
        print("Error reading input file: {}".format(e), file=sys.stderr)
        # Attempt to terminate workers gracefully
        for _ in range(args.parallel):
            try:
                input_queue.put(None, block=False)
            except Queue.Full:  # Use imported Queue exception
                pass
        sys.exit(1)
    except Exception as e:  # Catch other potential errors
        print("Unexpected error reading input file: {}".format(e), file=sys.stderr)
        sys.exit(1)

    # --- Signal end of input ---
    print("Finished reading input. Signaling workers to stop...")
    for _ in range(args.parallel):
        input_queue.put(None)  # Send sentinel value for each worker

    # --- Collect results ---
    print("Collecting results...")
    # Ensure total_lines is non-negative before collecting
    results = collect_results(output_queue, max(0, total_lines))

    # --- Wait for workers to finish ---
    print("Waiting for workers to terminate...")
    for p in workers:
        p.join(timeout=60)  # Add a timeout
        if p.is_alive():
            print(
                "Warning: Worker {} did not terminate gracefully. Forcing termination.".format(
                    p.pid
                ),
                file=sys.stderr,
            )
            p.terminate()
            p.join()

    # --- Calculate and print loss ---
    print("Calculating final loss...")
    final_loss, valid_count, fail_count, zero_true_count = calculate_loss(results)

    print("\n--- Results ---")
    print(
        "Total lines processed: {}".format(max(0, total_lines))
    )  # Ensure non-negative
    print("Successfully predicted (incl. zero true values): {}".format(valid_count))
    print("Failed/Skipped lines: {}".format(fail_count))
    print("Lines skipped due to zero true value: {}".format(zero_true_count))
    if final_loss is not None:
        # Use % formatting for Python 2
        print("Mean Absolute Percentage Error (MAPE): %.4f%%" % final_loss)
    else:
        print("Could not calculate loss (no valid non-zero true values).")


if __name__ == "__main__":
    # Setting start method isn't typically needed/available in Python 2 multiprocessing
    main()
