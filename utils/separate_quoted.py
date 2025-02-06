import csv
import re
import sys


def contains_quotes(text):
    """Checks if a string contains any quotes."""
    return '"' in text  # A simple check for any quote character


def process_tsv(input_file, quoted_output_file, non_quoted_output_file):
    """Reads a TSV file, filters sentences based on quotes, and writes to separate CSV files."""

    with open(input_file, "r", encoding="utf-8") as infile, open(
        quoted_output_file, "w", encoding="utf-8", newline=""
    ) as quoted_outfile, open(
        non_quoted_output_file, "w", encoding="utf-8", newline=""
    ) as non_quoted_outfile:

        reader = csv.reader(
            infile, delimiter="\x1f"
        )  # Correct delimiter for input
        quoted_writer = csv.writer(
            quoted_outfile, delimiter="\x1f"
        )  # Single tab delimiter for CSV
        non_quoted_writer = csv.writer(
            non_quoted_outfile, delimiter="\x1f"
        )  # Single tab delimiter for CSV

        for row in reader:
            if len(row) == 2:  # Ensure two columns
                en_text = row[0]
                kha_text = row[1]

                if contains_quotes(en_text) or contains_quotes(kha_text):
                    quoted_writer.writerow([en_text, kha_text])
                else:
                    non_quoted_writer.writerow([en_text, kha_text])
            else:
                print(f"Skipping row with unexpected number of columns: {row}")


# Example usage:
input_csv_file = sys.argv[1]
quoted_csv_file = sys.argv[2]
non_quoted_csv_file = sys.argv[3]

process_tsv(input_csv_file, quoted_csv_file, non_quoted_csv_file)

print(f"Sentences with quotes written to {quoted_csv_file}")
print(f"Sentences without quotes written to {non_quoted_csv_file}")
