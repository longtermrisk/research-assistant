import pytest
import os
import time
from pathlib import Path
import tempfile
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from automator.agent import Agent
from automator.workspace import Workspace


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests using real LLM calls.
    
    These tests require actual API keys and will make real API calls.
    Skip with: pytest -k "not integration"
    """
    
    def setup_method(self):
        # Create a temporary directory for workspaces
        self.temp_dir = tempfile.mkdtemp()
        # Set HOME to temp dir so ~/.automator points to the test directory
        self.old_home = os.environ.get("HOME")
        os.environ["HOME"] = self.temp_dir
        logger.info(f"Set up temp directory: {self.temp_dir}")
        
    def teardown_method(self):
        # Restore original HOME
        if self.old_home:
            os.environ["HOME"] = self.old_home
        # Clean up temp directory
        shutil.rmtree(self.temp_dir)
        logger.info("Cleaned up temp directory")
    
    @pytest.mark.asyncio
    async def test_simple_hello_world(self):
        """A simpler test just to check API connectivity."""
        logger.info("Starting simple hello world test")
        
        # Create a workspace and an agent
        workspace = Workspace("simple-test")
        agent = Agent(
            model="claude-3-5-haiku-20241022",
            prompt_template_yaml="prompts/chatgpt.yaml",
            tools=[],  # No tools needed for this test
            env={}
        )
        agent = workspace.add_agent(agent=agent, id="simple-agent")
        logger.info("Agent created")
        
        # Create a thread with a simple query
        thread = agent.run("Reply with 'Hello, World!'")
        logger.info("Thread created")
        
        # Process the thread once to get the response
        messages = []
        async for message in thread:
            messages.append(message)
            logger.info(f"Got message: {message}")
            break  # Just get the first response
        
        # Clean up
        await thread.cleanup()
        logger.info("Thread cleaned up")
        
        # Check the message
        assert len(messages) > 0, "No messages received"
        assert "Hello, World!" in str(messages[0].content), "Expected response not found"
        
        logger.info("Test completed successfully")

    @pytest.mark.asyncio
    async def test_terminal_tool_execution(self):
        """Test that an agent can use terminal tools to execute a simple command."""
        logger.info("Starting terminal tool execution test")
        
        # Create a workspace and an agent with terminal tools
        workspace = Workspace("test-workspace")
        logger.info("Workspace created")
        
        agent = Agent(
            model="claude-3-5-haiku-20241022",
            prompt_template_yaml="prompts/chatgpt.yaml",
            tools=[
                "terminal.terminal_execute",
                "terminal.terminal_stdin",
                "terminal.terminal_logs"
            ],
            env={}
        )
        agent = workspace.add_agent(agent=agent, id="test-agent")
        logger.info("Agent created with terminal tools")
        
        # Create a thread with instruction to run echo command
        thread = agent.run("Run this command: echo 'Hello, world!'")
        logger.info("Thread created with command request")
        
        # Process the thread and collect messages
        messages = []
        message_count = 0
        
        logger.info("Starting to process messages")
        async for message in thread:
            message_count += 1
            logger.info(f"Message {message_count} received, type: {message.role}")
            
            for i, content_block in enumerate(message.content):
                logger.info(f"  Content block {i}: {type(content_block).__name__}")
                if hasattr(content_block, 'type'):
                    logger.info(f"  Block type: {content_block.type}")
                    
                    # Log more details based on block type
                    if content_block.type == "tool_use":
                        logger.info(f"  Tool use: {content_block.name}")
                        logger.info(f"  Tool input: {content_block.input}")
                    elif content_block.type == "tool_result":
                        logger.info(f"  Tool result for: {content_block.tool_use_id}")
                        for j, result_block in enumerate(content_block.content):
                            if hasattr(result_block, 'text'):
                                logger.info(f"    Result text: {result_block.text[:100]}...")
            
            messages.append(message)
            
            # We need to break after a reasonable number of messages to avoid infinite loops
            if message_count >= 5:
                logger.info("Reached message limit, breaking loop")
                break
        
        # Clean up
        await thread.cleanup()
        logger.info("Thread cleaned up")
        
        # Check that messages were received
        assert len(messages) > 0, "No messages received"
        
        # Check for tool usage and results
        tool_use_found = False
        hello_world_found = False
        
        for message in messages:
            for content_block in message.content:
                # Check if the agent used the terminal tool
                if hasattr(content_block, 'type') and content_block.type == "tool_use":
                    if content_block.name == "terminal_execute":
                        tool_use_found = True
                        if "echo" in str(content_block.input) and "Hello, world" in str(content_block.input):
                            hello_world_found = True
                
                # Also check tool results for the output
                if hasattr(content_block, 'type') and content_block.type == "tool_result":
                    for result_block in content_block.content:
                        if hasattr(result_block, 'text') and "Hello, world" in result_block.text:
                            hello_world_found = True
        
        logger.info(f"Tool use found: {tool_use_found}")
        logger.info(f"Hello world found: {hello_world_found}")
        
        # Assertions
        assert tool_use_found, "Agent did not use terminal_execute tool"
        assert hello_world_found, "Hello, world! output not found in responses"
        
        logger.info("Test completed successfully")