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
			-At the end of your answer, list ONLY the source file names that you actually used.


		Retrieved Information:	

		Context: {context}

		Question: {question}

		Answer: """

	try:
		answer=ask_gemini(prompt)
	except Exception as e:
		return f"LLM Error: {str(e)}",[]	

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
			print("\nThank you for using Personal Second Brain!")
			break

		answer, sources=ask(question)
		
		print("\n"+"="*100)
		print("🧠 Personal Second Brain")
		print("="*100)

		print("\nQuery: ")
		print("-"*100)
		print(question)

		print("\nAnswer:")
		print("-"*100)
		print(answer)	
		print("-"*100)

		# print("\nSource File:")
		# for source in sources:
		# 	print(f"-{source}")	