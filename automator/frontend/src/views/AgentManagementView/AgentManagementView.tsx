import React, { useState, useEffect } from 'react';
import AppLayout from '../../components/AppLayout/AppLayout';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import { Agent, AgentCreatePayload } from '../../types';
import './AgentManagementView.css';

const AgentManagementView: React.FC = () => {
  const { currentWorkspace } = useWorkspace();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [showCreateForm, setShowCreateForm] = useState<boolean>(false);
  const [newAgent, setNewAgent] = useState<Partial<AgentCreatePayload>>({
    id: '',
    model: 'claude-3-haiku-20240307', // Default model
    prompt_template_yaml: 'prompts/chatgpt.yaml', // Default prompt
    tools: [],
    subagents: [],
  });

  const fetchAgents = async () => {
    if (!currentWorkspace) return;
    setIsLoading(true);
    setError(null);
    try {
      const fetchedAgents = await api.listAgents(currentWorkspace.name);
      setAgents(fetchedAgents);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch agents');
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchAgents();
  }, [currentWorkspace]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setNewAgent(prev => ({
      ...prev,
      [name]: name === 'tools' || name === 'subagents' ? value.split(',').map(s => s.trim()).filter(s => s) : value,
    }));
  };

  const handleCreateAgent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentWorkspace || !newAgent.id || !newAgent.model || !newAgent.prompt_template_yaml) {
      setError('Agent ID, Model, and Prompt Template YAML are required.');
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      // Ensure all fields are correctly typed for the payload
      const payload: AgentCreatePayload = {
        id: newAgent.id!,
        model: newAgent.model!,
        prompt_template_yaml: newAgent.prompt_template_yaml!,
        tools: newAgent.tools || [],
        env: newAgent.env || {},
        subagents: newAgent.subagents || [],
        as_tool: newAgent.as_tool || undefined, // Or null if API expects that for empty
        prompt_template_vars: newAgent.prompt_template_vars || undefined,
      };
      await api.createAgent(currentWorkspace.name, payload);
      setShowCreateForm(false);
      setNewAgent({ id: '', model: 'claude-3-haiku-20240307', prompt_template_yaml: 'prompts/chatgpt.yaml', tools: [], subagents: [] });
      fetchAgents(); // Refresh the list
    } catch (err: any) {
      setError(err.message || 'Failed to create agent');
    }
    setIsLoading(false);
  };

  return (
    <AppLayout>
      <div className="agent-management-container">
        <div className="header-actions">
          <h1>Agent Management</h1>
          <button onClick={() => setShowCreateForm(!showCreateForm)} className="button-primary">
            {showCreateForm ? 'Cancel' : '+ Create New Agent'}
          </button>
        </div>

        {isLoading && <p>Loading agents...</p>}
        {error && <p className="error-text">{error}</p>}

        {showCreateForm && (
          <form onSubmit={handleCreateAgent} className="create-agent-form">
            <h2>Create New Agent</h2>
            <div className="form-grid">
              <div className="form-group">
                <label htmlFor="id">Agent ID*</label>
                <input type="text" name="id" id="id" value={newAgent.id} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label htmlFor="model">Model*</label>
                <input type="text" name="model" id="model" value={newAgent.model} onChange={handleInputChange} required />
              </div>
              <div className="form-group full-width">
                <label htmlFor="prompt_template_yaml">Prompt Template YAML*</label>
                <input type="text" name="prompt_template_yaml" id="prompt_template_yaml" value={newAgent.prompt_template_yaml} onChange={handleInputChange} required />
              </div>
              <div className="form-group">
                <label htmlFor="tools">Tools (comma-separated, e.g., terminal.*,specific.tool)</label>
                <input type="text" name="tools" id="tools" value={(newAgent.tools || []).join(', ')} onChange={handleInputChange} />
              </div>
              <div className="form-group">
                <label htmlFor="subagents">Subagents (comma-separated IDs)</label>
                <input type="text" name="subagents" id="subagents" value={(newAgent.subagents || []).join(', ')} onChange={handleInputChange} />
              </div>
              {/* Add more fields for env, as_tool, prompt_template_vars if needed */}
            </div>
            <button type="submit" disabled={isLoading} className="button-primary submit-agent-btn">
              {isLoading ? 'Creating...' : 'Create Agent'}
            </button>
          </form>
        )}

        <div className="agents-list-container">
          <h2>Existing Agents</h2>
          {agents.length === 0 && !isLoading && (
            <p>No agents found in this workspace.</p>
          )}
          <ul className="agents-list">
            {agents.map(agent => (
              <li key={agent.id} className="agent-item">
                <h3>{agent.id}</h3>
                <p><strong>Model:</strong> {agent.model}</p>
                <p><strong>Prompt:</strong> {agent.prompt_template_yaml}</p>
                {agent.tools && agent.tools.length > 0 && (
                  <p><strong>Tools:</strong> {agent.tools.join(', ')}</p>
                )}
                {agent.subagents && agent.subagents.length > 0 && (
                  <p><strong>Subagents:</strong> {agent.subagents.join(', ')}</p>
                )}
                {/* Display other agent properties as needed */}
              </li>
            ))}
          </ul>
        </div>

      </div>
    </AppLayout>
  );
};

export default AgentManagementView;