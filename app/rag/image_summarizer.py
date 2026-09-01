import ollama
from PIL import Image
import tempfile
import os


def summarize_image(image_path):
	"""
		Summarize an image using Qwen2.5-VL
		Works for:
			-images contains text
			-images without text
			-images containing both text and visual information
	"""
	
	prompt="""
		Analyze this image and provide a clear, concise summary.

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
		- Only describe information that can be observed in the image.
	"""		

	response=ollama.chat(
		model="qwen2.5vl:3b",
		options={
			"num_ctx":8192
		},
		messages=[
			{
				"role":"user",
				"content":prompt,
				"images":[image_path]
			}
		]
	)

	return response['message']['content']