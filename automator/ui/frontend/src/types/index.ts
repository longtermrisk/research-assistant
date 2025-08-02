// Corresponds to MessageRole in dtypes.py
export enum MessageRole {
  User = "user",
  System = "system",
  Assistant = "assistant",
}

// Corresponds to Base64ImageSource in dtypes.py
export interface Base64ImageSource {
  data: string;
  media_type: string;
  type: "base64";
}

// Corresponds to various ContentBlock types in dtypes.py and ApiContentBlock in api/main.py
export interface TextBlock {
  type: "text";
  text: string;
  meta?: Record<string, any> | null;
}

export interface ImageBlock {
  type: "image";
  source: Base64ImageSource;
  meta?: Record<string, any> | null;
}

export interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input?: Record<string, any> | null;
  meta?: Record<string, any> | null;
}

export interface ToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content: ContentBlock[]; // Can contain TextBlock or ImageBlock
  meta?: Record<string, any> | null;
}

export type ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock;

// Corresponds to ApiChatMessage in api/main.py
export interface ApiChatMessage {
  role: MessageRole;
  content: ContentBlock[];
  meta?: Record<string, any> | null;
}

// Corresponds to WorkspaceResponse in api/main.py
export interface Workspace {
  name: string;
  path: string;
  env: Record<string, string>;
}

// Corresponds to AgentResponse in api/main.py
export interface Agent {
  id: string;
  model: string;
  prompt_template: string; // Changed from prompt_template_yaml based on AgentResponse in main.py
  tools: string[];
  env: Record<string, string>;
  subagents: string[];
  as_tool?: Record<string, any> | null;
  workspace_name: string;
  prompt_template_vars?: Record<string, any> | null;
}

// Corresponds to ThreadResponse in api/main.py
export interface ThreadSummary {
  id: string;
  model: string;
  tools: string[];
  env: Record<string, string>;
  subagents: string[];
  workspace_name: string;
  initial_messages_count: number; 
  first_user_message_preview?: string;
}

// Corresponds to ThreadDetailResponse in api/main.py
export interface ThreadDetail extends ThreadSummary {
  messages: ApiChatMessage[];
}

// For creating a workspace (corresponds to WorkspaceCreate)
export interface WorkspaceCreatePayload {
  name: string;
  env?: Record<string, string> | null;
}

// For creating an agent (corresponds to AgentCreate)
export interface AgentCreatePayload {
  id: string;
  model: string;
  prompt_template_yaml: string;
  tools?: string[] | null;
  env?: Record<string, string> | null;
  subagents?: string[] | null;
  as_tool?: Record<string, any> | null;
  prompt_template_vars?: Record<string, any> | null;
}

// For creating a thread (corresponds to ThreadCreateRequest on backend)
export interface ThreadCreatePayload {
  agent_id: string;
  initial_content: ContentBlock[]; // These are ApiContentBlock compatible
  thread_id?: string | null;
  mentioned_file_paths?: string[];
}

// For posting a message (corresponds to MessagePostRequest on backend)
export interface MessagePostPayload {
  content: ContentBlock[]; // These are ApiContentBlock compatible
  mentioned_file_paths?: string[];
}

// Definition for file/folder items in the workspace (corresponds to FileSystemItem)
export interface FileSystemItem {
  id: string; 
  name: string; 
  path: string; 
  type: 'file' | 'folder';
  children?: FileSystemItem[]; 
}

// Corresponds to ToolDefinition in dtypes.py (used by /tools endpoint)
export interface ToolDefinition {
  name: string;
  description?: string;
  input_schema?: Record<string, any> | null;
}

// Corresponds to McpServerTools in api/main.py (used by /tools endpoint)
export interface McpServerTools {
  server_name: string;
  tools: ToolDefinition[];
}