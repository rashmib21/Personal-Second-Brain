import hashlib


def calculate_file_hash(path):

    #object of sha256
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        chunk=file.read(8192) #read 8kb at a time

        while chunk:  
            sha256.update(chunk)
            chunk=file.read(8192)

    return sha256.hexdigest()