# 📋 RAG System Engineering Log — Problems & Resolutions Table

This log documents all technical problems faced across the system, detailing **What Happened** and **How We Fixed It** in clear human language with proper numbering.

| S. No. | Problem Faced | What Happened | How We Fixed It |
| :---: | :--- | :--- | :--- |
| **1** | **Chapter summarization was not working properly** | When I asked something like *“summarize chapter 1”*, the system treated it like a normal question-answering query instead of understanding that I wanted a summary of a complete section. | Added separate summarization intent detection so queries containing things like *chapter*, *section*, or *part* are handled by `search_for_summary()`. |
| **2** | **The system could not understand “SQL professionals”** | The actual file was named `SQLNotesForProfessionals.pdf`, so the wording in my query did not exactly match the filename. | Added token-based document matching and CamelCase splitting in [rag_pipeline.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/rag/rag_pipeline.py) so queries match split tokens of filenames. |
| **3** | **Hugging Face console deprecation warnings** | Initializing image processors without setting `use_fast` logged warnings about pure-Python vs fast C++/Rust backends. | Configured `tf_logging.set_verbosity_error()` and added `UserWarning` filters in [image_embedding.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/embeddings/image_embedding.py) to silence logs. |
| **4** | **Extra unrelated file sources appearing in image queries** | Asking *“What job roles are mentioned in the image?”* found `entry_level_jobs.webp`, but attached 9 extra unrelated PDF/Excel files due to a wide cutoff (`top_score - 4.5`). | Updated [vector_search.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/search/vector_search.py) to set a dynamic cutoff of `top_score - 1.0` when the top match is an image, dropping all unrelated text files. |
| **5** | **Unrelated personal queries (Salary) returning job spreadsheets** | Asking *“What is Rashmi's current salary?”* matched the word `"CTC"` in `Germany_IT_Companies.xlsx` because her resume contained no salary info and the cutoff floor `max(-2.0, ...)` clipped valid resume searches. | Adjusted general query rejection threshold to `-3.5`, added personal document checks in [vector_search.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/search/vector_search.py), and updated cutoff floor to `max(-8.0, top_score - 4.5)`. |
| **6** | **CrossEncoder GPU Out-Of-Memory (CUDA OOM) crash** | `CrossEncoder` attempted GPU memory allocations alongside active speech-to-text / PyTorch models, triggering a `torch.OutOfMemoryError` crash. | Set `reranker = CrossEncoder(..., device="cpu")` in [vector_search.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/search/vector_search.py). Reranking on CPU takes under 50ms and prevents GPU memory crashes. |
| **7** | **Loose CLIP distance image results in text queries** | General text questions called `search_images_by_text()` and received `entry_level_jobs.webp` with a loose CLIP distance (`0.72`), distorting text search candidate scoring. | Set distance threshold to `dist < 0.55` for general text queries and `dist < 0.85` for image queries in [vector_search.py](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/app/search/vector_search.py). |

---

### 📂 Document Downloads

* 📄 **Word Document Table (`.docx`):** [RAG_Engineering_Problems_Table.docx](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/docs/RAG_Engineering_Problems_Table.docx)
* 📝 **Markdown Document (`.md`):** [RAG_Engineering_Problems_Table.md](file:///home/rashmi/Data_Engineering/Personal%20Second%20Brain%20Project/docs/RAG_Engineering_Problems_Table.md)
