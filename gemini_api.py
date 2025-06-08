import os
from google import genai
from google.genai import types


def gemini_web_search(query: str, api_key: str) -> str:
    """
    Performs a web search using Google's Gemini model with search capabilities.

    Args:
        query: The search query.
        api_key: The Google AI API key.

    Returns:
        The search result as a string.

    Raises:
        ValueError: If the API key is missing.
        Exception: Propagates exceptions from the API call.
    """
    if not api_key:
        raise ValueError("API key for Gemini is missing.")

    try:
        client = genai.Client(api_key=api_key)

        model = "gemini-2.5-flash-preview-05-20"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=query),
                ],
            ),
        ]
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        generate_content_config = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )

        full_response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            full_response += chunk.text

        return full_response

    except Exception as e:
        print(f"Gemini web search failed with error: {e}")
        raise


if __name__ == "__main__":
    # For testing purposes
    # Make sure to set GEMINI_API_KEY_1 in your environment variables
    test_api_key = os.environ.get("GEMINI_API_KEY_1")
    if test_api_key:
        try:
            search_query = (
                "What are the latest developments in AI regulation in the EU?"
            )
            print(f"Performing test search for: '{search_query}'")
            result = gemini_web_search(search_query, test_api_key)
            print("\n--- Search Result ---")
            print(result)
            print("---------------------")
        except Exception as e:
            print(f"An error occurred during the test: {e}")
    else:
        print("GEMINI_API_KEY_1 environment variable not set. Skipping test.")
