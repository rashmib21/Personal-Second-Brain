import ollama 

MODEL_NAME="llama3.2"

def ask_llama(prompt):

	#Send a prompt to Ollama and return the response
	response=ollama.chat(
		model=MODEL_NAME,
		messages=[
			{
				"role":"user",
				"content":prompt,
			}])
	return response['message']['content']

if __name__=='__main__':
	print("="*100)
	print("Local AI Assistant (Llama 3.2)")
	print("Type 'exit' to quit")
	print("="*100)

	while True:
		user_input=input("\nYou: ").strip()
		if user_input.lower() in ['exit','quit']:
			print("\nGoodbye!")
			break
		if not user_input:
			continue
		try:
			answer=ask_llama(user_input)
			print("\nLlama: ", answer)
		except Exception as e:
			print(f"\nError: {e}")		
