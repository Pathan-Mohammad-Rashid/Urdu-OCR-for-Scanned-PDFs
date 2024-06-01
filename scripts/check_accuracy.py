# # import os

# # def check_and_correct_text(input_folder, output_folder):
# #     # Placeholder for text correction and accuracy improvement logic
# #     # Implement your own text correction algorithms or use external libraries
# #     for text_file in os.listdir(input_folder):
# #         with open(os.path.join(input_folder, text_file), 'r', encoding='utf-8') as file:
# #             text = file.read()
# #             corrected_text = text # Replace this with actual correction logic
            
# #         with open(os.path.join(output_folder, text_file), 'w', encoding='utf-8') as file:
# #             file.write(corrected_text)

# # if __name__ == "__main__":
# #     input_folder = 'extracted_text'
# #     output_folder = 'corrected_text'
# #     os.makedirs(output_folder, exist_ok=True)
# #     check_and_correct_text(input_folder, output_folder)





# import os

# def check_and_correct_text(input_folder, output_folder):
#     # Placeholder for text correction and accuracy improvement logic
#     # Implement your own text correction algorithms or use external libraries
#     for text_file in os.listdir(input_folder):
#         with open(os.path.join(input_folder, text_file), 'r', encoding='utf-8') as file:
#             text = file.read()
#             corrected_text = text # Replace this with actual correction logic
            
#         with open(os.path.join(output_folder, text_file), 'w', encoding='utf-8') as file:
#             file.write(corrected_text)

# if __name__ == "__main__":
#     input_folder = 'extracted_text'
#     output_folder = 'corrected_text'
#     os.makedirs(output_folder, exist_ok=True)
#     check_and_correct_text(input_folder, output_folder)


import os
from textblob import TextBlob

def correct_text(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())  # Corrects the text
    return corrected_text

def check_and_correct_text(input_folder, output_folder):
    for text_file in os.listdir(input_folder):
        with open(os.path.join(input_folder, text_file), 'r', encoding='utf-8') as file:
            text = file.read()
            corrected_text = correct_text(text)  # Using the correct_text function
            
        with open(os.path.join(output_folder, text_file), 'w', encoding='utf-8') as file:
            file.write(corrected_text)

if __name__ == "__main__":
    input_folder = 'extracted_text'
    output_folder = 'corrected_text'
    os.makedirs(output_folder, exist_ok=True)
    check_and_correct_text(input_folder, output_folder)
