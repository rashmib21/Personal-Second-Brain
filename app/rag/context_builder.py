def build_context(results):
	context=[]

	for doc in results:
		context.append(f"Source File: {doc['path']} Content: {doc['text']}")
	return "\n\n".join(context)	