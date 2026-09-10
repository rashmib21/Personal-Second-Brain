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
    Analyze an image using Qwen2.5-VL for Visual Question Answering (VQA).

    - Answers questions strictly from visual contents of image.
    - Never invents or guesses personal names / identities.
    """
    if not image_path or not os.path.exists(image_path):
        source_name = os.path.basename(image_path) if image_path else "the requested image"
        return f"No usable visual information was found for {source_name}."

    try:
        prepared_image, temporary_file = prepare_image_for_vlm(image_path)
    except Exception:
        source_name = os.path.basename(image_path)
        return f"No usable visual information was found for {source_name}."

    try:
        if question and question.strip():
            is_summary_or_text_req = any(w in question.lower() for w in [
                "summarize", "summarise", "summary", "describe", "overview",
                "text", "read", "transcribe", "writing", "content", "words", "letter",
                "inside", "say", "written", "code", "topic", "kafka", "redis"
            ])
            if is_summary_or_text_req:
                prompt = f"""You are a Vision Language Assistant.

Analyze the provided image carefully and fulfill the request. If the image contains handwritten or printed text, diagrams, or code, read and transcribe the visible information thoroughly.

User Request:
{question.strip()}

Rules:
- Provide a clear, accurate, and detailed answer reading all visible text, diagrams, code, titles, dates, numbers, and key facts.
- Do not invent information or names not present in the visual image.
- Do NOT guess personal names or personal identities for people unless written in text.

Response:"""
            else:
                prompt = f"""You are a Vision Language Assistant.

Analyze the provided image carefully and answer the following question based ONLY
on what is actually visible in the image.

User Question:
{question.strip()}

Rules:
- Answer ONLY the question asked.
- Do not provide unrelated information.
- Do not guess or invent information.
- Do NOT assign or guess personal names or personal identities (e.g., Rashmi). Describe visual features, counts, or clothing only.
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
- Identify the main objects, people count, scene, diagram, chart, or other relevant elements.

Rules:
- Do not invent information or personal identities.
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

    except Exception as e:
        source_name = os.path.basename(image_path)
        return f"No usable visual information was found for {source_name}."

    finally:

        if temporary_file and os.path.exists(temporary_file):
            try:
                os.remove(temporary_file)
            except Exception:
                pass