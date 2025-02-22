from pydantic import BaseModel
from openai import OpenAI

# Create an OpenAI client
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Option 1: Simple LLM call that returns plain text response
def ask_llm(question: str, model: str = "llama3.2") -> str:
    """
    Ask a question to the LLM and return the response as a text string.
    """
    chat_completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": question}
        ],
    )
    return chat_completion.choices[0].message.content

# Option 2: LLM call that returns structured JSON output parsed by a Pydantic model
def ask_llm_with_schema(question: str, response_model: type(BaseModel), model: str = "llama3.1:8b", temperature: float = 0):
    """
    Ask a question to the LLM and parse the JSON response using the provided Pydantic model.
    Parameters:
      - question: The input question to send.
      - response_model: A Pydantic model class that defines the expected response schema.
      - model: The model identifier to use.
      - temperature: The temperature setting for the LLM.
    Returns:
      - Parsed object if successful, or a refusal or error message if something goes wrong.
    """
    try:
        # Corrected beta endpoint access
        completion = client.chat.completions.create(
            temperature=temperature,
            model=model,
            messages=[
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object", "schema": response_model.model_json_schema()},
            # Add beta headers if required by your Ollama version
            extra_headers={"HTTP-Referer": "json-schema"}
        )
        result_message = completion.choices[0].message
        if result_message.content:
            # Manually parse the JSON content using the Pydantic model
            return response_model.parse_raw(result_message.content)
        else:
            return "No response content available"
    except Exception as e:
        return f"Error: {e}"

# --------------------------------------------------------------------------------------------------
# Usage Example
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Option 1: Basic text response
    print("=== Option 1: Simple Chat ===")
    response_text = ask_llm("Say this is a test", model="llama3.2")
    print(response_text)

    # Option 2: Structured response using Pydantic models
    # Define the expected response schema
    class FriendInfo(BaseModel):
        name: str
        age: int
        is_available: bool

    class FriendList(BaseModel):
        friends: list[FriendInfo]

    friend_question = (
        "I have two friends. The first is Ollama 22 years old busy saving the world, "
        "and the second is Alonso 23 years old and wants to hang out. "
        "Return a list of friends in JSON format with their name, age, and is_available status. "
        "is_available should be true if they want to hang out, false if they're busy."
    )
    print("\n=== Option 2: Structured Chat Response ===")
    structured_response = ask_llm_with_schema(friend_question, FriendList, model="llama3.2", temperature=0)
    print(structured_response)