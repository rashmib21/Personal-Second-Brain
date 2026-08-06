from google import genai
from config import GEMINI_API_KEY

client=genai.Client(api_key=GEMINI_API_KEY)
model_name="models/gemini-flash-latest"

def ask_gemini(prompt):
	response=client.models.generate_content(
		model=model_name,
		contents=prompt,
	)
	return response.text.strip()	