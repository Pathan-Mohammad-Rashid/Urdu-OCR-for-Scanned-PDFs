import os
from pdf2image import convert_from_path

def convert_pdfs_to_png(input_folder, output_folder):
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(input_folder, pdf_file)
        output_dir = os.path.join(output_folder, str(i + 1))
        os.makedirs(output_dir, exist_ok=True)
        
        pages = convert_from_path(pdf_path, dpi=300)
        for j, page in enumerate(pages):
            page.save(os.path.join(output_dir, f'page{j + 1}.png'), 'PNG')
    
if __name__ == "__main__":
    input_folder = 'books'
    output_folder = 'booksn'
    convert_pdfs_to_png(input_folder, output_folder)
