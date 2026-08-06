from app.search.vector_search import search
from app.llm.ollama_client import ask_llama
from app.llm.gemini_client import ask_gemini
import os

def ask(question):
	#Search similar chunks
	results=search(question)

	if not results:
		return "I could not find any relevant information in the indexed documents"

	#Combine all retrieved text
	context=[]

	for doc in results:
		context.append(
			f"Source File: {os.path.basename(doc['path'])} {doc['text']}")
	context="\n\n".join(context)
	
	#Make prompt
	prompt=f"""
		You are a Personal Second Brain Assistant.

		Answer the user's question using only the retrieved information below.

		Guidelines:
			-Use the retrieved document content and the document name ("Source file") if it helps answer the question.
			-Keep the answer concise and well structured.
			-If the answer cannot be determined from the retrieved information, reply exactly:
			"I could not find that information in the indexed documents."

		Retrieved Information:	

		Context: {context}

		Question: {question}

		Answer: """

	answer=ask_gemini(prompt)

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