import os
import sys
from pathlib import Path
from google import genai
from google.genai import types

# Add the parent directories to the path for relative imports
current_dir = Path(__file__).parent
models_dir = current_dir.parent / "models"
sys.path.append(str(models_dir))

from textsum_client import summarize_text_content
from youtube_client import summarize_youtube_video


def get_client_and_config():
    """
    Configure and return the Gemini client with function declarations.
    
    Returns:
        tuple: (client, config) for use with generate_content
    """
    
    # Get API key
    api_key = os.getenv("GOOGLE_GEMINI_KEY") 
    if not api_key:
        raise ValueError("No API key found. Set GOOGLE_GEMINI_KEY, GEMINI_API_KEY or AIPROXY_TOKEN environment variable.")
    
    # Create function declarations
    youtube_function = types.FunctionDeclaration(
        name="summarize_youtube_video",
        description="Summarizes a YouTube video by extracting and analyzing its content.",
        parameters={
            "type": "object",
            "properties": {
                "youtube_url": {
                    "type": "string",
                    "description": "The YouTube video URL to summarize",
                },
                "summary_length": {
                    "type": "string",
                    "description": "Length of summary. Can be: 1) Predefined: 'short' (50-100 words), 'medium' (150-250 words), 'long' (300-500 words), 2) Numeric: '100', '200', '300', 3) Text with units: '100 words', '200 words'. Defaults to 'medium' if not provided.",
                    "default": "medium"
                }
            },
            "required": ["youtube_url"],
        }
    )
    
    text_function = types.FunctionDeclaration(
        name="summarize_text_content",
        description="Summarizes provided text content with specified length.",
        parameters={
            "type": "object",
            "properties": {
                "text_content": {
                    "type": "string",
                    "description": "The text content to summarize",
                },
                "summary_length": {
                    "type": "string",
                    "description": "Length of summary. Can be: 1) Predefined: 'short' (50-100 words), 'medium' (150-250 words), 'long' (300-500 words), 2) Numeric: '100', '200', '300', 3) Text with units: '100 words', '200 words'. Defaults to 'medium' if not provided.",
                    "default": "medium"
                }
            },
            "required": ["text_content"],
        }
    )
    
    # Configure client and tools
    client = genai.Client(api_key=api_key)
    tools = types.Tool(function_declarations=[youtube_function, text_function])
    config = types.GenerateContentConfig(tools=[tools])
    
    return client, config


def register_functions():
    """Register actual function implementations for tool calling."""
    return {
        "summarize_youtube_video": summarize_youtube_video,
        "summarize_text_content": summarize_text_content
    }
