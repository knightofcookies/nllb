import csv
import re
import sys


def clean_quotes(text):
    """Removes excessive quotes from a string."""
    # Replace multiple consecutive quotes with a single quote
    text = re.sub(r'"{2,}', '"', text)
    # Remove leading and trailing quotes
    text = text.strip('"')
    return text


def process_tsv(input_file, output_file):
    """Reads a TSV file, cleans quotes, and writes to a new file."""
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.reader(
            infile, delimiter="\x1f"
        )  # Important: Specify the correct delimiter
        writer = csv.writer(
            outfile, delimiter="\x1f"
        )  # Use a single tab as delimiter for output

        for row in reader:
            if len(row) == 2:  # Ensure there are two columns
                en_cleaned = clean_quotes(row[0])
                kha_cleaned = clean_quotes(row[1])
                writer.writerow([en_cleaned, kha_cleaned])
            else:
                print(
                    f"Skipping row with unexpected number of columns: {row}"
                )  # Handle rows with incorrect format


if __name__ == "__main__":
    process_tsv(sys.argv[1], sys.argv[2])

    print(f"Cleaned data written to {sys.argv[2]}")
