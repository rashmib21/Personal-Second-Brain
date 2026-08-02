from app.search.vector_search import search
from app.llm.ollama_client import ask_llama

def ask(question):
	#Search similar chunks
	results=search(question)

	#Combine all retrieved text
	context=""

	for doc in results:
		context=context+doc.text
		context=context+"\n\n"

	#Make prompt
	prompt=f"""
		You are an AI assistant.

		Answer only using the context below.

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