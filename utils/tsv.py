# File paths for the input files and the output TSV file
en_file_path = 'en.txt'
kha_file_path = 'kha.txt'
output_tsv_path = 'parallel_corpus.tsv'

# Open the files and read their content
with open(en_file_path, 'r', encoding='utf-8') as en_file, \
     open(kha_file_path, 'r', encoding='utf-8') as kha_file:
    
    # Read the lines of both files
    en_lines = en_file.readlines()
    kha_lines = kha_file.readlines()
    
    # Ensure both files have the same number of lines
    if len(en_lines) != len(kha_lines):
        raise ValueError("The number of lines in both files does not match!")
    
    # Open the output TSV file for writing
    with open(output_tsv_path, 'w', encoding='utf-8') as tsv_file:
        # Write the header for the TSV file
        tsv_file.write('en\t\t\t\t\tkha\n')
        
        # Iterate through both files and write each pair to the TSV file
        for en_line, kha_line in zip(en_lines, kha_lines):
            # Strip any extra whitespace or newline characters
            en_line = en_line.strip()
            kha_line = kha_line.strip()
            
            # Write the English and Kha sentence to the TSV file
            tsv_file.write(f'{en_line}\t\t\t\t\t{kha_line}\n')

print(f"TSV file has been created: {output_tsv_path}")
