from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os


from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.agents.gemini_client import get_client_and_config, register_functions
from utils.models.textsum_client import summarize_text_content
from utils.models.youtube_client import summarize_youtube_video

app = FastAPI()

# Request models
class UserPrompt(BaseModel):
    prompt: str

class TextSummaryRequest(BaseModel):
    text_content: str
    summary_length: str = "medium"

class YouTubeSummaryRequest(BaseModel):
    youtube_url: str
    summary_length: str = "medium"

@app.get("/")
async def root():
    return {"message": "AI Agent API is running", "endpoints": ["/chat", "/summarize-text", "/summarize-youtube"]}

@app.post("/chat")
async def chat_with_agent(request: UserPrompt):
    """
    Main endpoint that processes user prompts using Gemini AI to determine
    the appropriate function (text summarization or YouTube video summarization)
    """
    try:
        # Get Gemini client and configuration
        client, config = get_client_and_config()
        
        # Register function implementations
        available_functions = register_functions()
        
        # Generate response using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=request.prompt,
            config=config
        )
        
        # Check if any function calls were made
        if response.candidates[0].content.parts:
            result_parts = []
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    # Execute the function call
                    function_name = part.function_call.name
                    function_args = dict(part.function_call.args)
                    
                    if function_name in available_functions:
                        try:
                            function_result = available_functions[function_name](**function_args)
                            result_parts.append({
                                "type": "function_result",
                                "function": function_name,
                                "args": function_args,
                                "result": function_result
                            })
                        except Exception as e:
                            result_parts.append({
                                "type": "function_error",
                                "function": function_name,
                                "error": str(e)
                            })
                    else:
                        result_parts.append({
                            "type": "unknown_function",
                            "function": function_name
                        })
                elif hasattr(part, 'text') and part.text:
                    result_parts.append({
                        "type": "text",
                        "content": part.text
                    })
            
            return {"response": result_parts}
        else:
            return {"response": [{"type": "text", "content": "No response generated"}]}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    

#Direct API for the function call 

# @app.post("/summarize-text")
# async def summarize_text_direct(request: TextSummaryRequest):
#     """
#     Direct endpoint for text summarization without using Gemini AI
#     """
#     try:
#         summary = summarize_text_content(request.text_content, request.summary_length)
#         return {"summary": summary, "input_length": len(request.text_content.split())}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


# @app.post("/summarize-youtube")
# async def summarize_youtube_direct(request: YouTubeSummaryRequest):
#     """
#     Direct endpoint for YouTube video summarization without using Gemini AI
#     """
#     try:
#         summary = summarize_youtube_video(request.youtube_url, request.summary_length)
#         return {"summary": summary, "video_url": request.youtube_url}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error summarizing YouTube video: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Agent API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)