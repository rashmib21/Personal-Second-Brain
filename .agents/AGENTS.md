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
