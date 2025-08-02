import asyncio
import os
import base64
from pydantic import BaseModel
from typing import List, Optional
from automator.dtypes import ChatMessage, MessageRole, TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, ToolDefinition
from automator.llm import get_response

# Test Pydantic model for structured output
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]

class WeatherInfo(BaseModel):
    location: str
    temperature: int
    condition: str
    humidity: Optional[int] = None

# Simple tool definition for testing
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is sunny and 72°F"

weather_tool = ToolDefinition(
    name="get_weather",
    description="Get current weather for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The location to get weather for"}
        },
        "required": ["location"]
    }
)

async def test_basic_functionality():
    """Test basic text generation"""
    print("=== Testing Basic Functionality ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text="Say hello in exactly 5 words.")])
    ]
    
    # Test with OpenAI if available
    if os.environ.get('OPENAI_API_KEY'):
        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                tools=None,
                max_tokens=50
            )
            print("OpenAI basic response:", response.content[0].text)
        except Exception as e:
            print(f"OpenAI basic error: {e}")
    
    # Test with Anthropic if available
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            response = await get_response(
                model="claude-3-haiku-20240307",
                messages=messages,
                tools=None,
                max_tokens=50
            )
            print("Anthropic basic response:", response.content[0].text)
        except Exception as e:
            print(f"Anthropic basic error: {e}")

async def test_structured_output():
    """Test structured output with Pydantic models"""
    print("\n=== Testing Structured Output ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="Alice and Bob are going to a science fair on Friday.")
        ])
    ]
    
    # Test with OpenAI if available
    if os.environ.get('OPENAI_API_KEY'):
        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                max_tokens=200
            )
            print("OpenAI structured response:", response)
            if hasattr(response, 'parsed') and response.parsed:
                print("Parsed event:", response.parsed)
        except Exception as e:
            print(f"OpenAI structured error: {e}")
    
    # Test with Anthropic (should raise NotImplementedError)
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            response = await get_response(
                model="claude-3-haiku-20240307",
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                max_tokens=200
            )
            print("Anthropic structured response:", response)
        except NotImplementedError as e:
            print(f"Anthropic structured (expected): {e}")
        except Exception as e:
            print(f"Anthropic structured error: {e}")

    if os.environ.get('GEMINI_API_KEY'):
        try:
            response = await get_response(
                model="gemini-2.5-flash",
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                max_tokens=200
            )
            print("Gemini structured response:", response)
            if hasattr(response, 'parsed') and response.parsed:
                print("Parsed event:", response.parsed)
        except Exception as e:
            print(f"Gemini structured error: {e}")

async def test_tool_usage():
    """Test tool usage"""
    print("\n=== Testing Tool Usage ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="What's the weather like in San Francisco?")
        ])
    ]
    
    tools = [weather_tool]
    
    # Test with OpenAI if available
    if os.environ.get('OPENAI_API_KEY'):
        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                max_tokens=200
            )
            print("OpenAI tool response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"OpenAI tool error: {e}")
    
    # Test with Anthropic if available
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            response = await get_response(
                model="claude-3-haiku-20240307",
                messages=messages,
                tools=tools,
                max_tokens=200
            )
            print("Anthropic tool response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"Anthropic tool error: {e}")

async def test_google_genai():
    """Test Google GenAI functionality"""
    print("\n=== Testing Google GenAI ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text="Hello, how are you?")])
    ]
    
    # Test with Google GenAI if available
    if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
        try:
            response = await get_response(
                model="gemini-2.5-flash",
                messages=messages,
                tools=None,
                max_tokens=50
            )
            print("Google GenAI response:", response.content[0].text[:100])
        except Exception as e:
            print(f"Google GenAI error: {e}")
    else:
        print("Google GenAI: No API key found")

async def main():
    """Run all tests"""
    # await test_basic_functionality()
    await test_structured_output()
    # await test_tool_usage()
    # await test_google_genai()

if __name__ == "__main__":
    asyncio.run(main())