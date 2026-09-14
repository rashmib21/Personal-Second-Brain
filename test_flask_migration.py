import os
import json
import urllib.parse
from app.server import app

def run_flask_migration_tests():
    print("==================================================")
    print("   FLASK API MIGRATION REGRESSION TEST SUITE    ")
    print("==================================================")

    client = app.test_client()

    # ----------------------------------------------------
    # TEST 1: System Stats Endpoint
    # ----------------------------------------------------
    print("\n--- TEST 1: GET /api/stats ---")
    res_stats = client.get("/api/stats")
    assert res_stats.status_code == 200, f"Expected 200, got {res_stats.status_code}"
    stats_data = res_stats.get_json()
    print("Status Code:", res_stats.status_code)
    print("Files Indexed:", stats_data.get("files_indexed"))
    print("Document Chunks:", stats_data.get("document_chunks"))
    print("Images Indexed:", stats_data.get("images_indexed"))
    print("Faces Indexed:", stats_data.get("faces_indexed"))

    # ----------------------------------------------------
    # TEST 2: File Inventory Endpoint
    # ----------------------------------------------------
    print("\n--- TEST 2: GET /api/files ---")
    res_files = client.get("/api/files")
    assert res_files.status_code == 200, f"Expected 200, got {res_files.status_code}"
    files_data = res_files.get_json()
    files_list = files_data.get("files", [])
    print(f"Total files returned: {len(files_list)}")
    if files_list:
        print("Sample file item:", files_list[0])

    # ----------------------------------------------------
    # TEST 3: Face Registry Endpoint
    # ----------------------------------------------------
    print("\n--- TEST 3: GET /api/faces ---")
    res_faces = client.get("/api/faces")
    assert res_faces.status_code == 200, f"Expected 200, got {res_faces.status_code}"
    faces_data = res_faces.get_json()
    print("Persons found:", faces_data.get("persons"))

    # ----------------------------------------------------
    # TEST 4: Audio Library Endpoint
    # ----------------------------------------------------
    print("\n--- TEST 4: GET /api/audio ---")
    res_audio = client.get("/api/audio")
    assert res_audio.status_code == 200, f"Expected 200, got {res_audio.status_code}"
    audio_data = res_audio.get_json()
    print(f"Total audio files: {len(audio_data.get('audio', []))}")

    # ----------------------------------------------------
    # TEST 5: Video Library Endpoint
    # ----------------------------------------------------
    print("\n--- TEST 5: GET /api/video ---")
    res_video = client.get("/api/video")
    assert res_video.status_code == 200, f"Expected 200, got {res_video.status_code}"
    video_data = res_video.get_json()
    print(f"Total video files: {len(video_data.get('videos', []))}")

    # ----------------------------------------------------
    # TEST 6: Normal Text QA Query
    # ----------------------------------------------------
    print("\n--- TEST 6: POST /api/chat ('what was discussed in the conference call?') ---")
    res_chat_qa = client.post("/api/chat", json={"question": "what was discussed in the conference call?"})
    assert res_chat_qa.status_code == 200, f"Expected 200, got {res_chat_qa.status_code}"
    qa_json = res_chat_qa.get_json()
    print("Answer snippet:", str(qa_json.get("answer"))[:200])
    print("Sources:", qa_json.get("sources"))

    # ----------------------------------------------------
    # TEST 7: Summary Query
    # ----------------------------------------------------
    print("\n--- TEST 7: POST /api/chat ('summarize my SQL notes') ---")
    res_chat_summary = client.post("/api/chat", json={"question": "summarize my SQL notes"})
    assert res_chat_summary.status_code == 200, f"Expected 200, got {res_chat_summary.status_code}"
    summary_json = res_chat_summary.get_json()
    print("Summary snippet:", str(summary_json.get("answer"))[:200])
    print("Sources:", summary_json.get("sources"))

    # ----------------------------------------------------
    # TEST 8: Image Query ("show the image of mummy")
    # ----------------------------------------------------
    print("\n--- TEST 8: POST /api/chat ('show the image of mummy') ---")
    res_chat_img1 = client.post("/api/chat", json={"question": "show the image of mummy"})
    assert res_chat_img1.status_code == 200, f"Expected 200, got {res_chat_img1.status_code}"
    img1_json = res_chat_img1.get_json()
    print("Answer:", img1_json.get("answer"))
    print("Images list:", img1_json.get("images"))
    print("Sources list:", img1_json.get("sources"))

    # Test media serving for returned image
    images_list = img1_json.get("images", [])
    if images_list:
        img_path = images_list[0]
        if isinstance(img_path, dict):
            img_path = img_path.get("path", "")
        encoded_img_path = urllib.parse.quote(img_path, safe="")
        media_url = f"/api/media/{encoded_img_path}"
        res_media = client.get(media_url)
        print(f"GET {media_url} Status Code: {res_media.status_code}")
        assert res_media.status_code == 200, f"Media serving failed for {img_path}"

    # ----------------------------------------------------
    # TEST 9: Paraphrased Image Query 1
    # ----------------------------------------------------
    print("\n--- TEST 9: Paraphrased Query 1 ('can you display my mummy image?') ---")
    res_chat_img2 = client.post("/api/chat", json={"question": "can you display my mummy image?"})
    assert res_chat_img2.status_code == 200, f"Expected 200, got {res_chat_img2.status_code}"
    img2_json = res_chat_img2.get_json()
    print("Images list:", img2_json.get("images"))

    # ----------------------------------------------------
    # TEST 10: Paraphrased Image Query 2
    # ----------------------------------------------------
    print("\n--- TEST 10: Paraphrased Query 2 ('please show me the mummy picture') ---")
    res_chat_img3 = client.post("/api/chat", json={"question": "please show me the mummy picture"})
    assert res_chat_img3.status_code == 200, f"Expected 200, got {res_chat_img3.status_code}"
    img3_json = res_chat_img3.get_json()
    print("Images list:", img3_json.get("images"))

    # ----------------------------------------------------
    # TEST 11: File Inventory Query
    # ----------------------------------------------------
    print("\n--- TEST 11: POST /api/chat ('give me my 3 newest files') ---")
    res_chat_inv = client.post("/api/chat", json={"question": "give me my 3 newest files"})
    assert res_chat_inv.status_code == 200, f"Expected 200, got {res_chat_inv.status_code}"
    inv_json = res_chat_inv.get_json()
    print("Answer snippet:", str(inv_json.get("answer"))[:200])

    # ----------------------------------------------------
    # TEST 12: Unknown / Ambiguous Topic Query
    # ----------------------------------------------------
    print("\n--- TEST 12: POST /api/chat ('tell me about quantum computing advancements in 2030') ---")
    res_chat_unk = client.post("/api/chat", json={"question": "tell me about quantum computing advancements in 2030"})
    assert res_chat_unk.status_code == 200, f"Expected 200, got {res_chat_unk.status_code}"
    unk_json = res_chat_unk.get_json()
    print("Answer snippet:", str(unk_json.get("answer"))[:200])

    print("\n==================================================")
    print("  ALL FLASK MIGRATION REGRESSION TESTS PASSED!    ")
    print("==================================================")

if __name__ == "__main__":
    run_flask_migration_tests()
