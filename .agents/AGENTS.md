# Coding Style Guidelines — Beginner Friendly

All frontend JavaScript/HTML/CSS code in this repository MUST follow these beginner-friendly guidelines:

1. **Simple and Readable over Short**: Readability > Understandability > Correct Functionality > Maintainability > Optimization.
2. **Explicit `if` / `else` Blocks**: Avoid single-line complex conditions or nested ternary operators (`a ? b : c ? d : e`). Use clear `if`, `else if`, and `else` blocks.
3. **Descriptive Variable Names**: Do not use single-letter variables like `e`, `q`, `f`, `d`, `res`. Use descriptive names like `event`, `searchQuery`, `file`, `responseData`, `response`.
4. **Normal Functions over Compact Callbacks**: Avoid complicated one-line arrow functions or chained array operations like `.map(...).join("")`. Use traditional loops or multi-line function bodies.
5. **Step-by-Step Logic**: Separate logical steps onto individual lines and include explanatory comments for each major step.
6. **Explicit State Handling**: Explicitly handle loading, success, and error states in API calls.
7. **No Unnecessary Operations per Line**: Keep one statement per line. Avoid compressing multiple operations or conditional execution on a single line (e.g. `if (condition) action();`).
8. **Interview-Friendly Code**: Structure and comment code so a student developer can easily read, explain, and defend every line during a technical interview.
9. **Automatic Git Push**: Whenever code or documentation changes are implemented and verified, ALWAYS create modular git commits and run `git push origin main` to ensure all changes are immediately published to the user's remote GitHub repository.

# Architectural & Generalization Guidelines — No Hardcoded Fixes

1. **Bug Reports as Reproduction Cases**: Never treat a reported query, word, filename, person, language, modality, or output as a special case to hardcode (`if query == ...`, `if filename == ...`, dictionary lookup for specific terms).
2. **Root Cause Analysis First**: Identify and fix the generic architectural layer (ingestion, ASR, translation, vector search, query parsing, RAG prompt, ordering) responsible for the bug.
3. **Source of Truth Principle**: Raw ASR transcript / original input is source evidence. Never silently alter, normalize, paraphrase, or infer intended meanings unless explicitly requested by user workflow.
4. **Generalization Requirement**: Every fix must work for unseen queries, unseen filenames, unseen languages, unseen speakers, and unseen data across all modalities.
5. **Pre-Modification Report Protocol**: Before modifying code for any bug fix, report:
   - ROOT CAUSE
   - AFFECTED COMPONENT
   - WHY CURRENT APPROACH FAILS
   - GENERAL FIX
   - WHY THIS GENERALIZES
   - FILES TO MODIFY / NOT MODIFY
   - REGRESSION TESTS
6. **Regression Testing**: Fixes must include tests covering: exact case, unseen variations, negative cases, different wordings/modalities.

