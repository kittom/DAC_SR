#!/usr/bin/env python3
"""
Script to format CSV data by removing headers and replacing commas with spaces.
"""

import os

def format_csv_to_txt():
    """
    Read the CSV file, remove the header, replace commas with spaces,
    and save as a text file.
    """
    # Define paths
    csv_path = "/home/mk422/Documents/DAC_SR/DataSets/Ground_Truth/GTLeadingOnes.csv"
    output_path = "/home/mk422/Documents/DAC_SR/SR_algorithms/Ai_feynman/GTLeadingOnes.txt"
    
    try:
        # Read the CSV file
        with open(csv_path, 'r') as csv_file:
            lines = csv_file.readlines()
        
        # Remove the first line (header) and process the data
        data_lines = lines[1:]  # Skip the header line
        
        # Replace commas with spaces for each line
        formatted_lines = []
        for line in data_lines:
            # Remove any trailing whitespace and replace commas with spaces
            formatted_line = line.strip().replace(',', ' ')
            formatted_lines.append(formatted_line)
        
        # Write the formatted data to the output file
        with open(output_path, 'w') as txt_file:
            for line in formatted_lines:
                txt_file.write(line + '\n')
        
        print(f"Successfully created {output_path}")
        print(f"Processed {len(formatted_lines)} data lines")
        
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {csv_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    format_csv_to_txt() 