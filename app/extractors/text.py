def extract_txt(path):
    """
    Extract text from a plain text file.
    """

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    return text