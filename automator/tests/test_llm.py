import asyncio
import os
import base64
from pydantic import BaseModel
from typing import List, Optional, Tuple
from automator.dtypes import ChatMessage, MessageRole, TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, ToolDefinition
from automator.llm import get_response


os.environ.pop('GOOGLE_API_KEY')
MAX_TOKENS=4000

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

def get_available_models() -> List[Tuple[str, str]]:
    """Check which API keys exist and return list of (model_name, provider) tuples"""
    models = []
    
    if os.environ.get('OPENAI_API_KEY'):
        models.append(("gpt-4o-mini", "OpenAI"))
    
    if os.environ.get('ANTHROPIC_API_KEY'):
        models.append(("claude-3-haiku-20240307", "Anthropic"))
    
    if os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'):
        models.append(("gemini-2.5-flash", "Google GenAI"))
    
    return models

def create_test_image() -> ImageBlock:
    """Create a simple test image (1x1 red pixel PNG)"""
    # This is a base64-encoded 1x1 red pixel PNG
    red_pixel_png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    return ImageBlock.from_base64(data=red_pixel_png, media_type="image/png")

async def test_basic_functionality():
    """Test basic text generation with all available models"""
    print("=== Testing Basic Functionality ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[TextBlock(text="How do I ask where is the library in spanish?")])
    ]
    
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=None,
                max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) basic response: {response.content[0].text}")
        except Exception as e:
            print(f"{provider} ({model_name}) basic error: {e}")

async def test_structured_output():
    """Test structured output with Pydantic models for all available models"""
    print("\n=== Testing Structured Output ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="Alice and Bob are going to a science fair on Friday.")
        ])
    ]
    
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=None,
                response_format=CalendarEvent,
                max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) structured response: {response}")
            if hasattr(response, 'parsed') and response.parsed:
                print(f"  Parsed event: {response.parsed}")
        except NotImplementedError as e:
            print(f"{provider} ({model_name}) structured (expected): {e}")
        except Exception as e:
            print(f"{provider} ({model_name}) structured error: {e}")

async def test_tool_usage():
    """Test tool usage with all available models"""
    print("\n=== Testing Tool Usage ===")
    
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="What's the weather like in San Francisco?")
        ])
    ]
    
    tools = [weather_tool]
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) tool response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text}")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"{provider} ({model_name}) tool error: {e}")

async def test_image_input():
    """Test image input with all available models"""
    print("\n=== Testing Image Input ===")
    
    test_image = create_test_image()
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            test_image,
            TextBlock(text="What do you see in this image? Please describe it briefly.")
        ])
    ]
    
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=None,
                max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) image response: {response.content[0].text[:150]}...")
        except Exception as e:
            print(f"{provider} ({model_name}) image error: {e}")

async def test_multimodal_with_tools():
    """Test combining images with tool usage"""
    print("\n=== Testing Multimodal with Tools ===")
    
    test_image = create_test_image()
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            test_image,
            TextBlock(text="Look at this image and then get the weather for New York.")
        ])
    ]
    
    tools = [weather_tool]
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            response = await get_response(
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=MAX_TOKENS
            )
            print(f"{provider} ({model_name}) multimodal+tools response:")
            for block in response.content:
                if isinstance(block, TextBlock):
                    print(f"  Text: {block.text[:100]}...")
                elif isinstance(block, ToolUseBlock):
                    print(f"  Tool call: {block.name}({block.input})")
        except Exception as e:
            print(f"{provider} ({model_name}) multimodal+tools error: {e}")

async def test_conversation_flow():
    """Test a multi-turn conversation with tool usage"""
    print("\n=== Testing Conversation Flow ===")
    
    models = get_available_models()
    if not models:
        print("No API keys found. Skipping tests.")
        return
    
    for model_name, provider in models:
        try:
            # First message: user asks for weather
            messages = [
                ChatMessage(role=MessageRole.user, content=[
                    TextBlock(text="What's the weather like in Boston?")
                ])
            ]
            
            # Get first response (should include tool call)
            response1 = await get_response(
                model=model_name,
                messages=messages,
                tools=[weather_tool],
                max_tokens=MAX_TOKENS
            )
            
            # Add the assistant's response to conversation
            messages.append(response1)
            
            # Simulate tool execution and add result
            tool_calls = [block for block in response1.content if isinstance(block, ToolUseBlock)]
            if tool_calls:
                tool_call = tool_calls[0]
                tool_result = ToolResultBlock(
                    tool_use_id=tool_call.id,
                    content=[TextBlock(text=get_weather(tool_call.input.get('location', 'Boston')))]
                )
                messages.append(ChatMessage(role=MessageRole.user, content=[tool_result]))
                
                # Get final response
                response2 = await get_response(
                    model=model_name,
                    messages=messages,
                    tools=[weather_tool],
                    max_tokens=MAX_TOKENS
                )
                
                print(f"{provider} ({model_name}) conversation flow:")
                print(f"  Tool call: {tool_call.name}({tool_call.input})")
                print(f"  Final response: {response2.content[0].text[:100]}...")
            else:
                print(f"{provider} ({model_name}) conversation flow: No tool call made")
                
        except Exception as e:
            print(f"{provider} ({model_name}) conversation flow error: {e}")

async def main():
    """Run all tests"""
    models = get_available_models()
    print(f"Found API keys for: {', '.join([f'{provider} ({model})' for model, provider in models])}")
    
    if not models:
        print("No API keys found. Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY")
        return
    
    await test_basic_functionality()
    await test_structured_output()
    await test_tool_usage()
    await test_image_input()
    await test_multimodal_with_tools()
    await test_conversation_flow()

if __name__ == "__main__":
    asyncio.run(main())