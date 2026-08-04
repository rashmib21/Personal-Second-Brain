from app.search.vector_search import search
from app.llm.ollama_client import ask_llama

def ask(question):
	#Search similar chunks
	results=search(question)

	if not results:
		return "I could not find any relevant information in the indexed documents"

	#Combine all retrieved text
	context=[]

	for doc in results:
		context.append(
			f"Source File: {doc['path']} {doc['text']}")
	context="\n\n".join(context)
	
	#Make prompt
	prompt=f"""
		You are an AI assistant.

		Use ONLY the information provided in the context below.

		Rules:
		1. Answer only from the provided context.
		2. If the answer is not found in the context, reply:
		   "I could not find that information in the indexed documents."
		3. Do not make up information.
		4. Mention the source file if possible.

		Context: {context}

		Question: {question}

		Answer: """

	return ask_llama(prompt)	

if __name__=="__main__":
	while True:
		question=input("\n\n\nAsk: ")
		if question.lower()=="exit":
			break
		answer=ask(question)
		
		print("\nAnswer:")
		print(answer)		