import re

def analyze_query(question):
	#Analyze the user query and decide which retrievel strategy to use, it doesn't perform retrievel
	query=question.strip().lower()

	# #Structural/section queries
	# structural_patterns=[
	# 	r"\bsummarize\s+(chapter|section|part)\b",
    #     r"\bsummarise\s+(chapter|section|part)\b",
    #     r"\bexplain\s+(chapter|section|part)\b",
    #     r"\bwhat is in\s+(chapter|section|part)\b",
    #     r"\bwhat does\s+(chapter|section|part)\b",
    # ]

    # for pattern in structural_patterns:
    # 	if re.search(pattern, query):
    # 		return {
    # 			"intent":"structural",
    # 			"retrieval_strategy":"structural",
    # 			"target":query,
    # 		}

    #Explicit section/heading style queries
	summary_patterns = [
		r"\bsummarize\b",
		r"\bsummarise\b",
		r"\bsummary\s+of\b",
		r"\bgive\s+(me\s+)?a\s+summary\b",
]

	for pattern in summary_patterns:
		if re.search(pattern,query):
			return {
				"intent":"summarization",
				"needs_database": True,
			}		

    #Everything else
	return {
		"intent": "question_answering",
		"needs_database": True
	}	