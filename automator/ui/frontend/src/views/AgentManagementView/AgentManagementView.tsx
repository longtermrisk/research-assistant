import React, { useState, useEffect } from 'react';
import AppLayout from '../../components/AppLayout/AppLayout';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api'; // Assuming api.ts has listModels & listAvailableTools
import { Agent, AgentCreatePayload, McpServerTools } from '../../types'; // Ensure McpServerTools is imported
import './AgentManagementView.css';

const AgentManagementView: React.FC = () => {
  const { currentWorkspace } = useWorkspace();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false); // General loading for agent list/creation
  const [error, setError] = useState<string | null>(null);

  const [showCreateForm, setShowCreateForm] = useState<boolean>(false);
  
  const initialDefaultModel = 'claude-3-7-sonnet-20250219';
  const initialNewAgentState: Partial<AgentCreatePayload> = {
    id: '',
    model: initialDefaultModel,
    prompt_template_yaml: 'chatgpt.yaml',
    tools: [],
    subagents: [],
    env: {},
    as_tool: undefined,
    prompt_template_vars: undefined,
  };
  const [newAgent, setNewAgent] = useState<Partial<AgentCreatePayload>>(initialNewAgentState);

  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(false);
  const [availableTools, setAvailableTools] = useState<string[]>([]); // Flattened list for select options
  const [isLoadingTools, setIsLoadingTools] = useState<boolean>(false);
  const [availablePrompts, setAvailablePrompts] = useState<string[]>([]);
  const [isLoadingPrompts, setIsLoadingPrompts] = useState<boolean>(false);

  const fetchAgents = async () => {
    if (!currentWorkspace) return;
    setIsLoading(true);
    setError(null);
    try {
      const fetchedAgents = await api.listAgents(currentWorkspace.name);
      setAgents(fetchedAgents);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch agents');
      console.error("Failed to fetch agents:", err);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    if (currentWorkspace) {
        fetchAgents();
    } else {
        setAgents([]); // Clear agents if no workspace is selected
    }
  }, [currentWorkspace]);

  useEffect(() => {
    if (showCreateForm) {
      // Fetch models
      const loadModels = async () => {
        setIsLoadingModels(true);
        try {
          const models = await api.listModels(); 
          setAvailableModels(models);
          // Set default model from fetched models if current newAgent.model is not in the list or not set
          if (models.length > 0) {
            if (!newAgent.model || !models.includes(newAgent.model)) {
                 setNewAgent(prev => ({ ...prev, model: models.includes(initialDefaultModel) ? initialDefaultModel : models[0] }));
            }
          } else {
            setNewAgent(prev => ({ ...prev, model: undefined })); // No models available
          }
        } catch (err: any) {
          console.error("Failed to fetch models:", err);
          setError(prevError => (prevError ? prevError + "\n" : "") + "Failed to fetch models.");
        }
        setIsLoadingModels(false);
      };

      // Fetch prompts
      const loadPrompts = async () => {
        setIsLoadingPrompts(true);
        try {
          const prompts = await api.listPrompts(); 
          setAvailablePrompts(prompts);
        } catch (err: any) {
          console.error("Failed to fetch prompts:", err);
          setError(prevError => (prevError ? prevError + "\n" : "") + "Failed to fetch prompts.");
        }
        setIsLoadingPrompts(false);
      };

      // Fetch tools
      const loadTools = async () => {
        setIsLoadingTools(true);
        try {
          const rawToolsData: McpServerTools[] = await api.listAvailableTools(); 
          const toolOptionsSet = new Set<string>();
          rawToolsData.forEach(serverTool => {
            // Add server-wide wildcard tool
            toolOptionsSet.add(`${serverTool.server_name}.*`);
            // Add specific tools
            serverTool.tools.forEach(tool => {
              toolOptionsSet.add(`${serverTool.server_name}.${tool.name}`);
            });
          });
          setAvailableTools(Array.from(toolOptionsSet).sort());
        } catch (err: any) {
          console.error("Failed to fetch tools:", err);
           setError(prevError => (prevError ? prevError + "\n" : "") + "Failed to fetch tools.");
        }
        setIsLoadingTools(false);
      };
      
      loadModels();
      loadTools();
      loadPrompts();
      // Agents for subagent dropdown are already available via `agents` state
    }
  }, [showCreateForm, currentWorkspace]); // Re-fetch if workspace changes while form is open

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setNewAgent(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleMultiSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name } = e.target;
    const selectedValues = Array.from(e.target.options)
      .filter(option => option.selected)
      .map(option => option.value);
    setNewAgent(prev => ({
      ...prev,
      [name]: selectedValues,
    }));
  };

  const handleCreateAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentWorkspace || !newAgent.id || !newAgent.model || !newAgent.prompt_template_yaml) {
      setError('Agent ID, Model, and Prompt Template YAML are required.');
      return;
    }
    setIsLoading(true); // Indicates creation is in progress
    setError(null);
    try {
      const payload: AgentCreatePayload = {
        id: newAgent.id!,
        model: newAgent.model!,
        prompt_template_yaml: newAgent.prompt_template_yaml!,
        tools: newAgent.tools || [],
        env: newAgent.env || {}, // Assuming env could be added later as a JSON textarea
        subagents: newAgent.subagents || [],
        as_tool: newAgent.as_tool || undefined, // Assuming as_tool could be added later
        prompt_template_vars: newAgent.prompt_template_vars || undefined, // Assuming prompt_template_vars could be added later
      };
      await api.createAgent(currentWorkspace.name, payload);
      setShowCreateForm(false);
      setNewAgent(initialNewAgentState); // Reset form to initial state
      fetchAgents(); // Refresh the list of agents
    } catch (err: any) {
      setError(err.message || 'Failed to create agent');
      console.error("Failed to create agent:", err);
    }
    setIsLoading(false); // Creation process finished
  };

  const subagentOptions = agents.map(agent => agent.id).sort();

  return (
    <AppLayout>
      <div className="agent-management-container">
        <div className="header-actions">
          <h1>Agent Management</h1>
          <button 
            onClick={() => {
              setShowCreateForm(!showCreateForm);
              if (!showCreateForm) { // If we are about to show the form
                setNewAgent(initialNewAgentState); // Reset to defaults
                setError(null); // Clear previous form-specific errors
              }
            }} 
            className="button-primary"
          >
            {showCreateForm ? 'Cancel' : '+ Create New Agent'}
          </button>
        </div>

        {/* Display general loading message for the agent list */}
        {isLoading && !showCreateForm && <p>Loading agents...</p>} 
        {error && <p className="error-text" style={{ whiteSpace: 'pre-line' }}>{error}</p>}

        {showCreateForm && (
          <form onSubmit={handleCreateAgent} className="create-agent-form">
            <h2>Create New Agent</h2>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="id">Agent ID*</label>
                <input type="text" name="id" id="id" value={newAgent.id || ''} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label htmlFor="model">Model*</label>
                <select 
                  name="model" 
                  id="model" 
                  value={newAgent.model || ''} 
                  onChange={handleInputChange} 
                  required 
                  disabled={isLoadingModels}
                >
                  {isLoadingModels && <option value="">Loading models...</option>}
                  {!isLoadingModels && availableModels.length === 0 && <option value="">No models available</option>}
                  {!isLoadingModels && availableModels.map(modelName => (
                    <option key={modelName} value={modelName}>{modelName}</option>
                  ))}
                </select>
              </div>
              <div className="form-group full-width">
                <label htmlFor="prompt_template_yaml">Prompt Template YAML*</label>
                <select 
                  name="prompt_template_yaml" 
                  id="prompt_template_yaml" 
                  value={newAgent.prompt_template_yaml || ''} 
                  onChange={handleInputChange} 
                  required
                >
                  <option value="">Select a prompt template</option>
                  {isLoadingPrompts && <option value="">Loading prompts...</option>}
                  {!isLoadingPrompts && availablePrompts.length === 0 && <option value="">No prompts available</option>}
                  {!isLoadingPrompts && availablePrompts.map(template => (
                    <option key={template} value={template}>{template}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="tools">Tools (Ctrl/Cmd+click for multiple)</label>
                <select 
                  multiple 
                  name="tools" 
                  id="tools" 
                  value={newAgent.tools || []} 
                  onChange={handleMultiSelectChange}
                  disabled={isLoadingTools}
                  size={5} // Show a few items
                >
                  {isLoadingTools && <option disabled>Loading tools...</option>}
                  {!isLoadingTools && availableTools.length === 0 && <option disabled>No tools available</option>}
                  {availableTools.map(toolName => (
                    <option key={toolName} value={toolName}>{toolName}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="subagents">Subagents (Ctrl/Cmd+click for multiple)</label>
                <select 
                  multiple 
                  name="subagents" 
                  id="subagents" 
                  value={newAgent.subagents || []} 
                  onChange={handleMultiSelectChange}
                  disabled={!currentWorkspace || agents.length === 0}
                  size={5} // Show a few items
                >
                  {(!currentWorkspace || agents.length === 0) && <option disabled>No other agents available in workspace</option>}
                  {subagentOptions.map(agentId => (
                    <option key={agentId} value={agentId}>{agentId}</option>
                  ))}
                </select>
              </div>
              {/* Placeholder for future fields:
              <div className="form-group full-width">
                <label htmlFor="env">Environment Variables (JSON)</label>
                <textarea name="env" id="env" value={newAgent.env ? JSON.stringify(newAgent.env) : '{}'} onChange={handleJsonInputChange} />
              </div>
              */}
            </div>
            <button type="submit" disabled={isLoading || isLoadingModels || isLoadingTools} className="button-primary submit-agent-btn">
              {isLoading ? 'Creating...' : 'Create Agent'}
            </button>
          </form>
        )}

        <div className="agents-list-container">
          <h2>Existing Agents</h2>
          {/* Message when agent list is loading is handled by the general isLoading above */}
          {agents.length === 0 && !isLoading && (
            <p>No agents found in this workspace. {currentWorkspace ? '' : 'Please select a workspace first.'}</p>
          )}
          <ul className="agents-list">
            {agents.map(agent => (
              <li key={agent.id} className="agent-item">
                <h3>{agent.id}</h3>
                <p><strong>Model:</strong> {agent.model}</p>
                {/* In AgentResponse, prompt_template is the resolved prompt, not the yaml path */}
                <p><strong>Prompt Template:</strong> {agent.prompt_template}</p> 
                {agent.tools && agent.tools.length > 0 && (
                  <p><strong>Tools:</strong> {agent.tools.join(', ')}</p>
                )}
                {agent.subagents && agent.subagents.length > 0 && (
                  <p><strong>Subagents:</strong> {agent.subagents.join(', ')}</p>
                )}
                {/* Display other agent properties as needed, e.g., env, as_tool */}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </AppLayout>
  );
};

export default AgentManagementView;