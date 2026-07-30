# Responsible for extracting text from Excel (.xlsx) files

from openpyxl import load_workbook


def extract_spreadsheet(path):

    # Open the workbook
    workbook = load_workbook(path)

    # Store extracted text
    text = ""

    # Visit every sheet
    for sheet in workbook.worksheets:

        # Visit every row
        for row in sheet.iter_rows(values_only=True):

            # Visit every cell
            for cell in row:

                # Ignore empty cells
                if cell is not None:

                    # Add cell value
                    text = text + str(cell) + " "

            # Move to next line
            text = text + "\n"

    # Return extracted text
    return text


# # Testing
# if __name__ == "__main__":
#
#     file_path = "watched_folder/sample.xlsx"
#
#     text = extract_spreadsheet(file_path)
#
#     print(text)