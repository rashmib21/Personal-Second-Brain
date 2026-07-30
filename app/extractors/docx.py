# Responsible for extracting text from Microsoft Word (.docx) files

from docx import Document


def extract_docx(path):

    # Open the Word document
    document = Document(path)

    # Store extracted text
    text = ""

    # Visit every paragraph
    for paragraph in document.paragraphs:

        # Add paragraph text
        text = text + paragraph.text + "\n"

    # Return extracted text
    return text


# # Testing
# if __name__ == "__main__":
#
#     file_path = "watched_folder/sample.docx"
#
#     text = extract_docx(file_path)
#
#     print(text)