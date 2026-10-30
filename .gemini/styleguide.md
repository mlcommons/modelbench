# Python Style Guide

## Key Principles

*   **Readability:** Code should be easy to understand for all team members.
*   **Maintainability:** Code should be easy to modify and extend.
*   **Consistency:** Adhering to a consistent style across all projects
    improves collaboration and reduces errors.
*   **Performance:** While readability is paramount, code should be efficient.

## Deviations from PEP 8

### Line Length

* **Maximum line length:** 120 characters (instead of PEP 8's 79).

### Imports

*   **Import order within groups:**  Sort alphabetically.

### Naming Conventions

*   **Variables:** Use lowercase with underscores (snake_case): `user_name`, `total_count`
*   **Constants:**  Use uppercase with underscores: `MAX_VALUE`, `DATABASE_NAME`
*   **Functions:** Use lowercase with underscores (snake_case): `calculate_total()`, `process_data()`
*   **Classes:** Use CapWords (CamelCase): `UserManager`, `PaymentProcessor`
*   **Modules:** Use lowercase with underscores (snake_case): `user_utils`, `payment_gateway`

### Type Hints

*   **Use type hints:**  Type hints improve code readability and help catch
    errors early.
*   **Follow PEP 484:**  Use the standard type hinting syntax.

### Comments

*   **Write clear and concise comments:** Explain the "why" behind the code,
    not just the "what".
*   **Comment sparingly:** Well-written code should be self-documenting where
    possible.

### Error Handling

*   **Use specific exceptions:** Avoid using broad exceptions like `Exception`.
*   **Handle exceptions gracefully:** Provide informative error messages.
*   **Use `try...except` blocks and/or context managers:**  Isolate code that might raise exceptions.

## Tooling

*   **Code formatter:** black
