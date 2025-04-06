import os
import csv


def save_dict_of_lists_to_csv(data, output_path):
    """
    Merges new data with existing CSV columns (side-by-side) instead of appending rows.

    Parameters:
        data (dict): Dictionary where keys are column names and values are lists of data.
        output_path (str): Path to the output CSV file.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Step 1: Load existing CSV data (if available)
    existing_data = {}
    if os.path.exists(output_path):
        with open(output_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            existing_data = {col: [] for col in reader.fieldnames}

            # Iterate over rows and fill the dictionary with column data
            for row in reader:
                for col in reader.fieldnames:
                    existing_data[col].append(row[col])

    # Step 2: Merge existing columns with new data
    all_headers = set(existing_data.keys()).union(data.keys())

    merged_data = {
        header: existing_data.get(header, []) + data.get(header, [])
        for header in all_headers
    }

    # Step 3: Normalize column lengths
    # max_length = max(len(lst) for lst in merged_data.values())
    # for key in merged_data:
    #     merged_data[key] += [""] * (max_length - len(merged_data[key]))  # Fill missing values

    # Step 4: Write updated data back to CSV
    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(merged_data.keys())  # Headers
        writer.writerows(zip(*merged_data.values()))  # Row-wise writing

    print(f"Data saved to {output_path}")


def save_to_summary(data):
    output_file = os.path.join("results/metrics", "data_summary.csv")
    save_dict_of_lists_to_csv(data, output_file)
