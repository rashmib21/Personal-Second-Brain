from app.rag.image_summarizer import summarize_image

image_path = "/home/rashmi/Downloads/test2.jpg"

summary = summarize_image(image_path)

print("\n===== IMAGE SUMMARY =====")
print(summary)