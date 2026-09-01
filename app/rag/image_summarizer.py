import ollama
from PIL import Image
import tempfile
import os


def prepare_image_for_vlm(image_path, max_size=768):

	"""
	Prepare an image for Qwen2.5-VL.

	- Small images are used directly.
	- Large images are resized temporarily.
	- Original image is never modified.
	"""

	image = Image.open(image_path)

	# If image is already small enough, use original
	if image.width <= max_size and image.height <= max_size:
		return image_path, None

	# Create resized copy
	image.thumbnail((max_size, max_size))

	# Temporary file
	temp_file = tempfile.NamedTemporaryFile(
		suffix=".jpg",
		delete=False
	)
	temp_path = temp_file.name
	temp_file.close()

	# Convert to RGB because JPEG doesn't support some image modes
	if image.mode != "RGB":
		image = image.convert("RGB")

	image.save(temp_path, format="JPEG", quality=85)

	return temp_path, temp_path


def summarize_image(image_path, question=None):
    """
    Analyze an image using Qwen2.5-VL.

    - If a question is provided, answer that question.
    - If no question is provided, generate a general image summary.
    """

    prepared_image, temporary_file = prepare_image_for_vlm(image_path)

    try:
        if question and question.strip():

            prompt = f"""You are a Vision Language Assistant.

Analyze the provided image carefully and answer the following question based ONLY
on what is actually visible in the image.

User Question:
{question.strip()}

Rules:
- Answer ONLY the question asked.
- Do not provide unrelated information.
- Do not guess or invent information.
- If the requested information is not visible in the image, say:
  "Information not found in the image."
- For YES/NO questions, answer ONLY:
  YES
  or
  NO

Answer:"""

        else:

            prompt = """Analyze this image and provide a clear, concise summary.

If the image contains text:
- Read the important text.
- Include important names, titles, numbers, dates, and facts.

If the image contains no text:
- Describe the important visual content.
- Identify the main objects, people, scene, diagram, chart, or other relevant elements.

If the image contains both text and visual information:
- Combine the important textual and visual information.

Rules:
- Do not invent information.
- Only describe information that can actually be observed in the image.
"""

        response = ollama.chat(
            model="qwen2.5vl:3b",
            options={
                "num_ctx": 4096,
                "num_gpu": 0
            },
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [prepared_image]
                }
            ]
        )

        return response["message"]["content"].strip()

    finally:

        if temporary_file and os.path.exists(temporary_file):
            try:
                os.remove(temporary_file)
            except Exception:
                pass