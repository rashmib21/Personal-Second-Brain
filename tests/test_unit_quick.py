import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.query.query_analyzer import analyze_query, find_best_matching_source, extract_entity_tokens
from app.rag.rag_pipeline import handle_metadata_query, is_filename_stem_match


def run_unit_quick():
    print("=" * 80)
    print("RUNNING QUICK UNIT TESTS FOR NEW ARCHITECTURE")
    print("=" * 80)

    # 1. Test possessive and typo-tolerant entity token extraction
    tokens1 = extract_entity_tokens("what are the name of jethlal 's friends?")
    print("Tokens for 'jethlal 's':", tokens1)
    assert "jethlal" in tokens1, "FAILED: 'jethlal' not in tokens!"

    # 2. Test typo-tolerant source resolution
    indexed_files = [
        "Jethalal ने Liye गरम गरम Jalebi Fafda के मज़े !  TMKOC Movies  Taarak Mehta Ka Ooltah Chashmah  - Taarak Mehta ka Ooltah Chashmah Movies.mp3",
        "Behari_lal_call.m4a",
        "Accounts.m4a",
        "dense.jpeg",
        "dense2.webp",
        "dense3.webp",
        "mummy.jpg"
    ]

    matched_src, conf = find_best_matching_source("what are the name of jethlal 's friends?", indexed_files)
    print(f"Matched source for 'jethlal 's': {matched_src} (confidence: {conf})")
    assert matched_src and "Jethalal" in matched_src and conf >= 0.70, "FAILED: Typo resolution for Jethalal failed!"

    matched_behari, conf_behari = find_best_matching_source("how many chunks you have for the file behari lal?", indexed_files)
    print(f"Matched source for 'behari lal': {matched_behari} (confidence: {conf_behari})")
    assert matched_behari == "Behari_lal_call.m4a", "FAILED: Behari lal resolution failed!"

    # 3. Test Unresolved Explicit Source Detection (Bug #2)
    analysis_unresolved = analyze_query("what are the names in non_existent_file_xyz?", indexed_files)
    print("Unresolved source analysis:", analysis_unresolved)
    assert analysis_unresolved.get("unresolved_explicit_source") == True, "FAILED: Unresolved explicit source not detected!"
    assert str(analysis_unresolved.get("source_hint")).startswith("UNRESOLVED_"), "FAILED: UNRESOLVED_ prefix missing!"

    # 4. Test Dense Image Filename Queries (Bug #3)
    analysis_dense = analyze_query("please list the name images which have named by dense.", indexed_files)
    print("Dense query intent:", analysis_dense.get("intent"))
    print("Dense entity tokens:", extract_entity_tokens("please list the name images which have named by dense."))
    assert analysis_dense.get("intent") == "IMAGE_FILENAME_QUERY", "FAILED: Expected IMAGE_FILENAME_QUERY!"

    res_meta = handle_metadata_query("please list the name images which have named by dense.", analysis_dense)
    matched_imgs = res_meta[1] if isinstance(res_meta, tuple) else []
    print("Matched dense images:", matched_imgs)
    assert set(matched_imgs) == {"dense.jpeg", "dense2.webp", "dense3.webp"}, f"FAILED: Expected exact dense images, got {matched_imgs}"

    # 5. Test Hinglish Intent Classification (Bug #5)
    analysis_hinglish = analyze_query("con call me kya baatein hui hai?", indexed_files)
    print("Hinglish query intent:", analysis_hinglish.get("intent"))
    print("Hinglish query source:", analysis_hinglish.get("source_hint"))
    assert analysis_hinglish.get("intent") == "AUDIO_SUMMARY", "FAILED: Expected AUDIO_SUMMARY intent for Hinglish call query!"
    assert analysis_hinglish.get("source_hint") == "Behari_lal_call.m4a", "FAILED: Hinglish query failed to resolve Behari lal audio!"

    # 6. Test Stem Matching Logic for dense variants
    assert is_filename_stem_match("dense", "dense.jpeg") == True
    assert is_filename_stem_match("dense", "dense2.webp") == True
    assert is_filename_stem_match("dense", "dense_2.png") == True
    assert is_filename_stem_match("dense", "mummy.jpg") == False

    print("\n" + "=" * 80)
    print("ALL QUICK UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_unit_quick()
