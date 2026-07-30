# Responsible for extracting file names from ZIP archives

import zipfile


def extract_archive(path):

    # Store extracted information
    text = ""

    # Open the ZIP archive
    with zipfile.ZipFile(path, "r") as archive:

        # Visit every file inside the archive
        for file_name in archive.namelist():

            # Add file name to the text
            text = text + file_name + "\n"

    # Return all file names
    return text


# # Testing
# if __name__ == "__main__":
#
#     file_path = "watched_folder/sample.zip"
#
#     text = extract_archive(file_path)
#
#     print(text)	