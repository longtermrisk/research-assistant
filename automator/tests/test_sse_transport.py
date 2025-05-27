"""
Tests for SSE transport support in the automator package.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from automator.agent import Thread
from automator.dtypes import ChatMessage, MessageRole, TextBlock

@pytest.fixture
def mock_servers_config():
    """Mock server configuration with both stdio and SSE transports."""
    return {
        "stdio-server": {
            "command": "python",
            "args": ["server.py"],
            "env": {"KEY": "value"}
        },
        "sse-server": {
            "transport": "sse",
            "url": "https://example.com/sse",
            "headers": {"Authorization": "Bearer token"},
            "timeout": 10,
            "sse_read_timeout": 300,
            "env": {"API_TOKEN": "test-token"}
        }
    }

@pytest.mark.asyncio
async def test_stdio_transport_backward_compatibility(mock_servers_config):
    """Test that stdio transport still works as before."""
    with patch('automator.agent._SERVERS', mock_servers_config):
        with patch('automator.agent.stdio_client') as mock_stdio_client:
            # Mock the stdio client context manager
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)
            
            # Mock ClientSession
            with patch('automator.agent.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                thread = Thread(
                    model="test-model",
                    messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])],
                    tools=["stdio-server.*"],
                    env={}
                )
                
                await thread.prepare()
                
                # Verify stdio_client was called with correct parameters
                mock_stdio_client.assert_called_once()
                call_args = mock_stdio_client.call_args[0][0]
                assert call_args.command == "python"
                assert call_args.args == ["server.py"]
                assert call_args.env == {"KEY": "value"}

@pytest.mark.asyncio
async def test_sse_transport_configuration(mock_servers_config):
    """Test that SSE transport is properly configured."""
    with patch('automator.agent._SERVERS', mock_servers_config):
        with patch('automator.agent.sse_client') as mock_sse_client:
            # Mock the SSE client context manager
            mock_read = AsyncMock()
            mock_write = AsyncMock()
            mock_sse_client.return_value.__aenter__.return_value = (mock_read, mock_write)
            
            # Mock ClientSession
            with patch('automator.agent.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session.initialize = AsyncMock()
                mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                mock_session_class.return_value.__aenter__.return_value = mock_session
                
                thread = Thread(
                    model="test-model",
                    messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])],
                    tools=["sse-server.*"],
                    env={"EXTRA_TOKEN": "extra-value"}
                )
                
                await thread.prepare()
                
                # Verify sse_client was called with correct parameters
                mock_sse_client.assert_called_once()
                call_kwargs = mock_sse_client.call_args[1]
                
                assert call_kwargs['url'] == "https://example.com/sse"
                assert call_kwargs['timeout'] == 10
                assert call_kwargs['sse_read_timeout'] == 300
                
                # Check headers include both configured headers and auto-converted env vars
                headers = call_kwargs['headers']
                assert headers['Authorization'] == "Bearer token"
                assert 'X-API-TOKEN' in headers
                assert headers['X-API-TOKEN'] == "test-token"
                assert 'X-EXTRA-TOKEN' in headers
                assert headers['X-EXTRA-TOKEN'] == "extra-value"

@pytest.mark.asyncio
async def test_sse_transport_missing_url():
    """Test that SSE transport raises error when URL is missing."""
    mock_config = {
        "bad-sse-server": {
            "transport": "sse"
            # Missing URL
        }
    }
    
    with patch('automator.agent._SERVERS', mock_config):
        thread = Thread(
            model="test-model",
            messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])],
            tools=["bad-sse-server.*"],
            env={}
        )
        
        with pytest.raises(ValueError, match="SSE transport requires 'url'"):
            await thread.prepare()

@pytest.mark.asyncio
async def test_unknown_transport_type():
    """Test that unknown transport types raise an error."""
    mock_config = {
        "unknown-transport-server": {
            "transport": "websocket"  # Not supported
        }
    }
    
    with patch('automator.agent._SERVERS', mock_config):
        thread = Thread(
            model="test-model",
            messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])],
            tools=["unknown-transport-server.*"],
            env={}
        )
        
        with pytest.raises(ValueError, match="Unknown transport type 'websocket'"):
            await thread.prepare()

@pytest.mark.asyncio
async def test_mixed_transports(mock_servers_config):
    """Test that stdio and SSE transports can be used together."""
    with patch('automator.agent._SERVERS', mock_servers_config):
        with patch('automator.agent.stdio_client') as mock_stdio_client:
            with patch('automator.agent.sse_client') as mock_sse_client:
                # Mock both clients
                mock_stdio_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
                mock_sse_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
                
                # Mock ClientSession
                with patch('automator.agent.ClientSession') as mock_session_class:
                    mock_session = AsyncMock()
                    mock_session.initialize = AsyncMock()
                    mock_session.list_tools = AsyncMock(return_value=Mock(tools=[]))
                    mock_session_class.return_value.__aenter__.return_value = mock_session
                    
                    thread = Thread(
                        model="test-model",
                        messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text="test")])],
                        tools=["stdio-server.*", "sse-server.*"],
                        env={}
                    )
                    
                    await thread.prepare()
                    
                    # Verify both transports were initialized
                    mock_stdio_client.assert_called_once()
                    mock_sse_client.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])