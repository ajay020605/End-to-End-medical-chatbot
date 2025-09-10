
# Updated Prompt to match the required Gemini format
prompt_template = (
    "Context: {context}\n\n"
    "Input: {input}\n\n"
    "Provide me the response from the context according to the input from the user. "
    "The response should be formal and clear, maintaining the wording style of the user's input. "
    "Ensure the response is strictly from the context, declarative, and follows the user's tone. "
    "Modify the given context accordingly."
)