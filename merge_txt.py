import os


def merge_text_files(input_directory, output_file):
    txt_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]
    txt_files.sort()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for txt_file in txt_files:
            file_path = os.path.join(input_directory, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        # Remove leading slash if present and write to output file
                        cleaned_line = line.lstrip('/')
                        outfile.write(cleaned_line)
            except Exception as e:
                print(f"Error reading {txt_file}: {str(e)}")

    print(f"Successfully merged {len(txt_files)} files into {output_file}")


if __name__ == "__main__":
    input_dir = "SUN397/test_label"
    output_file = "SUN397_TestImages.txt"
    merge_text_files(input_dir, output_file)
