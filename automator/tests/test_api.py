"""
Tests for the Backend API
"""
import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_workspace_endpoints():
    """Test the workspace endpoints"""
    # Create a workspace
    response = client.post(
        "/workspaces",
        json={"name": "Test Workspace"}
    )
    assert response.status_code == 200
    workspace_data = response.json()
    assert workspace_data["name"] == "Test Workspace"
    workspace_id = workspace_data["id"]
    
    # Get the workspace
    response = client.get(f"/workspaces/{workspace_id}")
    assert response.status_code == 200
    assert response.json()["id"] == workspace_id
    
    # List workspaces
    response = client.get("/workspaces")
    assert response.status_code == 200
    workspaces = response.json()
    assert any(w["id"] == workspace_id for w in workspaces)

def test_agent_endpoints():
    """Test the agent endpoints"""
    # Create a workspace first
    response = client.post(
        "/workspaces",
        json={"name": "Agent Test Workspace"}
    )
    workspace_id = response.json()["id"]
    
    # Create an agent
    response = client.post(
        f"/workspaces/{workspace_id}/agents",
        json={
            "name": "Test Agent",
            "model": "claude-3-5-haiku-20241022",
            "prompt_template_yaml": "prompts/chatgpt.yaml",
            "tools": ["terminal.terminal_logs"]
        }
    )
    assert response.status_code == 200
    agent_data = response.json()
    assert agent_data["name"] == "Test Agent"
    agent_id = agent_data["id"]
    
    # Get the agent
    response = client.get(f"/workspaces/{workspace_id}/agents/{agent_id}")
    assert response.status_code == 200
    assert response.json()["id"] == agent_id
    
    # List agents
    response = client.get(f"/workspaces/{workspace_id}/agents")
    assert response.status_code == 200
    agents = response.json()
    assert any(a["id"] == agent_id for a in agents)

def test_utility_endpoints():
    """Test the utility endpoints"""
    # List prompts
    response = client.get("/prompts")
    assert response.status_code == 200
    
    # List tools
    response = client.get("/tools")
    assert response.status_code == 200
    
    # List models
    response = client.get("/models")
    assert response.status_code == 200