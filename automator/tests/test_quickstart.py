"""Test based on automator/examples/quickstart.py"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
from localrouter import TextBlock

pytestmark = pytest.mark.asyncio


async def test_agent_with_terminal_date_query():
    """Test creating an agent with terminal tool and asking for the date."""
    from automator.agent import Agent
    from automator.workspace import Workspace
    
    # Use a temporary workspace to avoid conflicts
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up temporary home directory for this test
        original_home = os.environ.get('HOME')
        try:
            os.environ['HOME'] = tmpdir
            
            # Create minimal prompts directory structure
            prompts_dir = Path(tmpdir) / '.automator' / 'prompts'
            prompts_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the assistant.yaml prompt file we need
            assistant_yaml_content = """messages:
  - role: system
    content: |
      You are a helpful assistant that can use tools to help users.
  - role: user
    content: $query
"""
            (prompts_dir / 'assistant.yaml').write_text(assistant_yaml_content)
            
            # Create workspace and agent
            workspace = Workspace('test-quickstart')
            agent = Agent(
                llm={"model": "gemini-2.5-pro"},
                prompt_template_yaml="assistant.yaml",
                tools=["terminal.*"]
            )
            agent = workspace.add_agent(agent=agent, id="bash")
            
            # Test the agent by asking for the date
            query = "What is the current date? Please use the 'date' command."
            thread = await agent.run(query)
            
            # Collect all messages from the thread
            messages = []
            async for message in thread:
                messages.append(message)
            
            # Verify we got a response
            assert len(messages) > 0, "No messages received from agent"
            
            # The agent should have made at least one tool call to terminal
            found_terminal_call = False
            found_date_response = False
            
            for message in messages:
                for block in message.content:
                    if hasattr(block, 'name') and 'terminal' in block.name.lower():
                        found_terminal_call = True
                    elif isinstance(block, TextBlock) and any(date_word in block.text.lower() for date_word in ['date', 'today', '2025']):
                        found_date_response = True
            
            assert found_terminal_call, "Agent did not make a terminal tool call"
            assert found_date_response, "Agent response did not contain date information"
            
            await thread.cleanup()
            
        finally:
            # Restore original HOME environment
            if original_home:
                os.environ['HOME'] = original_home
            elif 'HOME' in os.environ:
                del os.environ['HOME']


async def test_basic_agent_creation():
    """Test basic agent creation without execution."""
    from automator.agent import Agent
    from automator.workspace import Workspace
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_home = os.environ.get('HOME')
        try:
            os.environ['HOME'] = tmpdir
            
            # Create minimal prompts directory structure
            prompts_dir = Path(tmpdir) / '.automator' / 'prompts'
            prompts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a simple assistant.yaml
            assistant_yaml_content = """messages:
  - role: system
    content: |
      You are a helpful assistant.
  - role: user
    content: $query
"""
            (prompts_dir / 'assistant.yaml').write_text(assistant_yaml_content)
            
            # Test agent creation
            workspace = Workspace('test-basic')
            agent = Agent(
                llm={"model": "gemini-2.5-pro"},
                prompt_template_yaml="assistant.yaml",
                tools=["terminal.*"]
            )
            
            # Verify agent was created successfully
            assert agent is not None
            assert agent.llm["model"] == "gemini-2.5-pro"
            
            # Add agent to workspace
            agent = workspace.add_agent(agent=agent, id="test-agent")
            assert agent is not None
            
        finally:
            if original_home:
                os.environ['HOME'] = original_home
            elif 'HOME' in os.environ:
                del os.environ['HOME']