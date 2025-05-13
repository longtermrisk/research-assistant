// Basic API service structure
import {
  Workspace, WorkspaceCreatePayload,
  Agent, AgentCreatePayload,
  ThreadSummary, ThreadDetail, ThreadCreatePayload, MessagePostPayload,
  ApiChatMessage,
  FileSystemItem,
  McpServerTools // Added McpServerTools
} from '../types';

const API_BASE_URL = 'http://localhost:8000'; // Assuming backend runs on port 8000

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// --- Global Endpoints ---
export async function listModels(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/models`);
  return handleResponse<string[]>(response);
}

export async function listAvailableTools(): Promise<McpServerTools[]> {
  const response = await fetch(`${API_BASE_URL}/tools`);
  return handleResponse<McpServerTools[]>(response);
}

// --- Workspace Endpoints ---
export async function createWorkspace(payload: WorkspaceCreatePayload): Promise<Workspace> {
  const response = await fetch(`${API_BASE_URL}/workspaces`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<Workspace>(response);
}

export async function listWorkspaces(): Promise<Workspace[]> {
  const response = await fetch(`${API_BASE_URL}/workspaces`);
  return handleResponse<Workspace[]>(response);
}

export async function getWorkspaceDetails(workspaceName: string): Promise<Workspace> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}`);
  return handleResponse<Workspace>(response);
}

export async function listWorkspaceFiles(workspaceName: string): Promise<FileSystemItem[]> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/files`);
  return handleResponse<FileSystemItem[]>(response);
}


// --- Agent Endpoints ---
export async function createAgent(workspaceName: string, payload: AgentCreatePayload): Promise<Agent> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/agents`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<Agent>(response);
}

export async function listAgents(workspaceName: string): Promise<Agent[]> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/agents`);
  return handleResponse<Agent[]>(response);
}

export async function getAgentDetails(workspaceName: string, agentId: string): Promise<Agent> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/agents/${agentId}`);
  return handleResponse<Agent>(response);
}

// --- Thread Endpoints ---
export async function createThread(workspaceName: string, payload: ThreadCreatePayload): Promise<ThreadSummary> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/threads`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<ThreadSummary>(response);
}

export async function listThreads(workspaceName: string): Promise<ThreadSummary[]> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/threads`);
  return handleResponse<ThreadSummary[]>(response);
}

export async function getThreadDetails(workspaceName: string, threadId: string): Promise<ThreadDetail> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/threads/${threadId}`);
  return handleResponse<ThreadDetail>(response);
}

export async function postMessage(workspaceName: string, threadId: string, payload: MessagePostPayload): Promise<ApiChatMessage> {
  const response = await fetch(`${API_BASE_URL}/workspaces/${workspaceName}/threads/${threadId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return handleResponse<ApiChatMessage>(response);
}

// SSE connection helper - not a direct API call, but related
export function getMessagesSSE(workspaceName: string, threadId: string): EventSource {
  return new EventSource(`${API_BASE_URL}/workspaces/${workspaceName}/threads/${threadId}/messages/sse`);
}