from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
import os

def ask(question):
	#Search similar chunks
	results=search(question)

	if not results:
		return "I could not find any relevant information in the indexed documents",[]

	#Combine all retrieved text
	context=[]

	for doc in results:
		context.append(
			f"Source File: {doc['path']} {doc['text']}")
	context="\n\n".join(context)
	
	#Make prompt
	prompt=f"""
		You are an AI assistant.

		Answer only using the context below.

		Context: {context}

		Question: {question}

		Answer: """

	answer=ask_llama(prompt)

	#collect unique source files
	sources=[]
	for doc in results:
		filename=os.path.basename(doc['path'])
		if filename not in sources:
			sources.append(filename)

	return answer,sources	

if __name__=="__main__":
	while True:
		question=input("\n\n\nAsk: ")
		if question.lower()=="exit":
			break
		answer, sources=ask(question)
		
		print("\nAnswer:")
		print(answer)	

		print("\nSource File:")
		for source in sources:
			print(f"-{source}")	