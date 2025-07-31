import asyncio
import os
from automator.dtypes import ChatMessage, MessageRole, TextBlock, ToolDefinition
from automator.llm import get_response

# Test tool for weather
weather_tool = ToolDefinition(
    name="get_weather",
    description="Get current weather information for any location worldwide",
    input_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string", 
                "description": "The location to get weather for (city, country)"
            }
        },
        "required": ["location"]
    }
)

async def test_gemini_tools():
    """Test Google GenAI tool usage specifically"""
    if not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found")
        return
    
    print("🧪 Testing Gemini Tool Usage")
    print("=" * 40)
    
    # Test 1: Simple tool call
    print("\n1️⃣ Simple tool call test:")
    messages = [
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="What's the weather like in Tokyo? Use the get_weather tool.")
        ])
    ]
    
    try:
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[weather_tool],
            max_tokens=200
        )
        
        print("Response content:")
        tool_calls_found = False
        for i, block in enumerate(response.content):
            if hasattr(block, 'text'):
                print(f"  Text {i+1}: {block.text}")
            elif hasattr(block, 'name'):
                print(f"  Tool Call {i+1}: {block.name}({block.input})")
                tool_calls_found = True
        
        if tool_calls_found:
            print("✅ Tool calls detected - working correctly!")
        else:
            print("❌ No tool calls found - tools may not be passed correctly")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: With system instruction
    print("\n2️⃣ Tool call with system instruction:")
    messages = [
        ChatMessage(role=MessageRole.system, content=[
            TextBlock(text="You are a helpful weather assistant. Always use the get_weather tool when asked about weather.")
        ]),
        ChatMessage(role=MessageRole.user, content=[
            TextBlock(text="What's the weather in London?")
        ])
    ]
    
    try:
        response = await get_response(
            model="gemini-2.5-flash",
            messages=messages,
            tools=[weather_tool],
            max_tokens=200
        )
        
        print("Response content:")
        tool_calls_found = False
        for i, block in enumerate(response.content):
            if hasattr(block, 'text'):
                print(f"  Text {i+1}: {block.text}")
            elif hasattr(block, 'name'):
                print(f"  Tool Call {i+1}: {block.name}({block.input})")
                tool_calls_found = True
        
        if tool_calls_found:
            print("✅ Tool calls with system instruction working!")
        else:
            print("❌ No tool calls found with system instruction")
            
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Compare with OpenAI (if available)
    if os.environ.get('OPENAI_API_KEY'):
        print("\n3️⃣ Comparison with OpenAI:")
        try:
            response = await get_response(
                model="gpt-4o-mini",
                messages=[ChatMessage(role=MessageRole.user, content=[
                    TextBlock(text="What's the weather in Paris? Use the get_weather tool.")
                ])],
                tools=[weather_tool],
                max_tokens=200
            )
            
            print("OpenAI response:")
            for i, block in enumerate(response.content):
                if hasattr(block, 'text'):
                    print(f"  Text {i+1}: {block.text}")
                elif hasattr(block, 'name'):
                    print(f"  Tool Call {i+1}: {block.name}({block.input})")
                    
        except Exception as e:
            print(f"❌ OpenAI Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_tools())