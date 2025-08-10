#!/usr/bin/env python3
"""Test script to verify environment variable resolution in MCP config"""

import os
import json
import tempfile
from pathlib import Path

# Test the environment variable resolution
def test_env_resolution():
    # Create a mock mcp.json with environment variable references
    mock_mcp_config = {
        "mcpServers": {
            "test_server": {
                "command": "test_command", 
                "args": ["arg1"],
                "env": {
                    "OPENAI_API_KEY": "$OPENAI_API_KEY",
                    "STATIC_VAR": "static_value",
                    "MISSING_VAR": "$MISSING_VAR"
                }
            }
        }
    }
    
    # Set up test environment
    os.environ["OPENAI_API_KEY"] = "test_key_123"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary mcp.json
        mcp_file = Path(tmpdir) / "mcp.json"
        with open(mcp_file, "w") as f:
            json.dump(mock_mcp_config, f)
        
        # Update HOME to point to temp directory
        original_home = os.environ.get('HOME')
        os.environ['HOME'] = tmpdir
        
        try:
            # Import and test the thread's resolve_env_vars method
            from automator.agent import Thread
            
            thread = Thread(llm={"model": "gemini-2.5-pro"})
            
            # Test environment resolution
            env_config = {
                "OPENAI_API_KEY": "$OPENAI_API_KEY",
                "STATIC_VAR": "static_value", 
                "MISSING_VAR": "$MISSING_VAR"
            }
            
            agent_env = {"AGENT_VAR": "agent_value"}
            
            resolved = thread.resolve_env_vars(env_config, agent_env)
            
            print("Original env config:", env_config)
            print("Resolved env config:", resolved)
            
            # Verify resolution worked correctly
            assert resolved["OPENAI_API_KEY"] == "test_key_123", f"Expected 'test_key_123', got {resolved['OPENAI_API_KEY']}"
            assert resolved["STATIC_VAR"] == "static_value", f"Static value should remain unchanged"
            assert resolved["MISSING_VAR"] == "$MISSING_VAR", f"Missing variable should remain as-is"
            
            print("✅ Environment variable resolution test passed!")
            
        finally:
            # Restore original HOME
            if original_home:
                os.environ['HOME'] = original_home
            else:
                os.environ.pop('HOME', None)

if __name__ == "__main__":
    test_env_resolution()