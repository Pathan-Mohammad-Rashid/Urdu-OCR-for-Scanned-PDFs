# Urdu OCR for Scanned PDFs

This repository provides a solution for extracting Urdu text from scanned PDFs using OCR techniques, specifically leveraging the UTRNet Model. The workflow converts PDF pages into high-quality PNG images, applies the UTRNet model to recognize and extract text from these images, and saves the extracted text into a structured format.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
The `urdu_ocr_for_scanned_pdfs` repository automates the process of extracting text from Urdu PDFs. The key steps include:
1. Converting PDF pages to high-quality PNG images.
2. Applying the UTRNet model to these images to perform OCR.
3. Saving the extracted text into organized text files.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Pathan-Mohammad-Rashid/urdu_LLM.git
cd urdu_LLM
pip install -r requirements.txt
```

Ensure you have the necessary libraries and tools for PDF and image processing, as well as the UTRNet model.

## Usage
The main script, `app.py`, handles the entire workflow. 

1. Place your PDF files in the `books` folder.
2. Run the script:

```bash
python app.py
```

The script will:
- Convert each PDF page in the `books` folder to a high-quality PNG image.
- Use the UTRNet model to extract text from the images saved in the `pdf_images` folder.
- Save the extracted text into the `extracted_text` folder, with each book's text stored in a separate file (`bookn.txt`).

## File Structure
- `books/`: Folder to store PDF files to be processed.
- `pdf_images/`: Folder where converted PNG images from PDFs will be saved.
- `extracted_text/`: Folder where the extracted text files will be stored (`book1.txt`, `book2.txt`, etc.).
- `app.py`: Main script to perform the OCR process.
- `requirements.txt`: List of required dependencies.

## Citation
If you use the code/dataset, please cite the following paper:
```bash
@InProceedings{10.1007/978-3-031-41734-4_19,
		author="Rahman, Abdur
		and Ghosh, Arjun
		and Arora, Chetan",
		editor="Fink, Gernot A.
		and Jain, Rajiv
		and Kise, Koichi
		and Zanibbi, Richard",
		title="UTRNet: High-Resolution Urdu Text Recognition in Printed Documents",
		booktitle="Document Analysis and Recognition - ICDAR 2023",
		year="2023",
		publisher="Springer Nature Switzerland",
		address="Cham",
		pages="305--324",
		isbn="978-3-031-41734-4",
		doi="https://doi.org/10.1007/978-3-031-41734-4_19"
}
```

## Contributing

Contributions are welcome! Please create a pull request with detailed information on what changes you are proposing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to explore the repository and enhance the OCR capabilities for Urdu PDFs. Happy coding!
