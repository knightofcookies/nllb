# File paths for the input files and the output TSV file
en_file_path = '../datasets/europarl-v7.fr-en.en'
fr_file_path = '../datasets/europarl-v7.fr-en.fr'
output_tsv_path = '../datasets/europarl_fr_en_parallel_corpus.tsv'

# Open the files and read their content
with open(en_file_path, 'r', encoding='utf-8') as en_file, \
     open(fr_file_path, 'r', encoding='utf-8') as fr_file:
    
    # Read the lines of both files
    en_lines = en_file.readlines()
    fr_lines = fr_file.readlines()
    
    # Ensure both files have the same number of lines
    if len(en_lines) != len(fr_lines):
        raise ValueError("The number of lines in both files does not match!")
    
    # Open the output TSV file for writing
    with open(output_tsv_path, 'w', encoding='utf-8') as tsv_file:
        # Write the header for the TSV file
        tsv_file.write('en\t\t\t\t\tfr\n')
        
        # Iterate through both files and write each pair to the TSV file
        for en_line, fr_line in zip(en_lines, fr_lines):
            # Strip any extra whitespace or newline characters
            en_line = en_line.strip()
            fr_line = fr_line.strip()
            
            # Write the English and fr sentence to the TSV file
            tsv_file.write(f'{en_line}\t\t\t\t\t{fr_line}\n')

print(f"TSV file has been created: {output_tsv_path}")
