import sys
from app.server import app

client = app.test_client()

print("=== TESTING FLASK WEB SERVER ENDPOINTS ===")

# 1. Test /api/stats
res_stats = client.get("/api/stats")
print(f"\n1. GET /api/stats [{res_stats.status_code}]:")
print(res_stats.get_json())

# 2. Test /api/files
res_files = client.get("/api/files")
print(f"\n2. GET /api/files [{res_files.status_code}]:")
print(res_files.get_json())

# 3. Test /api/faces
res_faces = client.get("/api/faces")
print(f"\n3. GET /api/faces [{res_faces.status_code}]:")
print(res_faces.get_json())

# 4. Test /api/audio
res_audio = client.get("/api/audio")
print(f"\n4. GET /api/audio [{res_audio.status_code}]:")
print(res_audio.get_json())

# 5. Test /api/chat with person image query
res_chat1 = client.post("/api/chat", json={"question": "can you show me the picture of rashmi ?"})
print(f"\n5. POST /api/chat ('show me picture of rashmi') [{res_chat1.status_code}]:")
data1 = res_chat1.get_json()
print("ANSWER:\n", data1.get("answer"))
print("IMAGES:\n", data1.get("images"))
print("SOURCES:\n", data1.get("sources"))

# 6. Test /api/chat with audio MLK speech query
res_chat2 = client.post("/api/chat", json={"question": "What does Martin Luther King Jr. say about the Emancipation Proclamation?"})
print(f"\n6. POST /api/chat ('MLK speech') [{res_chat2.status_code}]:")
data2 = res_chat2.get_json()
print("ANSWER:\n", data2.get("answer"))
print("SOURCES:\n", data2.get("sources"))

print("\n=== ALL FLASK SERVER ENDPOINTS TESTED SUCCESSFULLY ===")
