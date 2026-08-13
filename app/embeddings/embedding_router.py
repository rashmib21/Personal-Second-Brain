from app.embeddings.text_embedding import embed_text
from app.embeddings.image_embedding import embed_image

def generate_embedding(file_type, content):
	# Strip leading dot if present (e.g. '.txt' -> 'txt')
	file_type = str(file_type).lstrip(".").lower()

	#Text based
	if file_type in (
    # Documents
    "pdf",
    "doc",
    "docx",
    "odt",
    "rtf",
    "tex",

    # Text / Data
    "txt",
    "text",
    "md",
    "markdown",
    "rst",
    "csv",
    "tsv",
    "json",
    "xml",
    "yaml",
    "yml",
    "log",

    # Spreadsheets
    "xls",
    "xlsx",
    "xlsm",
    "xlsb",
    "ods",
    "spreadsheet",

    # Presentations
    "ppt",
    "pptx",
    "pptm",
    "odp",
    "presentation",

    # Audio
    "mp3",
    "wav",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "oga",
    "opus",
    "wma",
    "aiff",
    "aif",
    "amr",
    "audio",

    # Video
    "mp4",
    "avi",
    "mov",
    "mkv",
    "wmv",
    "flv",
    "webm",
    "mpeg",
    "mpg",
    "m4v",
    "3gp",
    "3g2",
    "mts",
    "m2ts",
    "ts",
    "vob",
    "ogv",
    "video",

    # Internal / normalized types
    "document",
    "text",
    "spreadsheet",
    "presentation",
    "audio",
    "video",
    "video_transcript",
):
		return embed_text(content)

	#Image based 
	elif file_type in ("jpg","jpeg","png", "image","video_frame",):
		return embed_image(content)

	raise ValueError(f"Unsupported file type: {file_type}")	
