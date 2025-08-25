import React, { useState, useEffect } from 'react';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import { Workspace } from '../../types';
import './WorkspaceSelectionView.css'; // We'll create this CSS file next

const WorkspaceSelectionView: React.FC = () => {
  const { selectWorkspace, createAndSelectWorkspace, isLoading, error } = useWorkspace();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [newWorkspaceName, setNewWorkspaceName] = useState<string>('');
  const [isLoadingList, setIsLoadingList] = useState<boolean>(false);
  const [listError, setListError] = useState<string | null>(null);

  useEffect(() => {
    fetchWorkspaces();
  }, []);

  const fetchWorkspaces = async () => {
    setIsLoadingList(true);
    setListError(null);
    try {
      const fetchedWorkspaces = await api.listWorkspaces();
      setWorkspaces(fetchedWorkspaces);
    } catch (err: any) {
      setListError(err.message || 'Failed to fetch workspaces');
    }
    setIsLoadingList(false);
  };

  const handleSelectWorkspace = (workspaceName: string) => {
    selectWorkspace(workspaceName);
  };

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWorkspaceName.trim()) return;
    await createAndSelectWorkspace(newWorkspaceName.trim());
    // Optionally, refresh the list or optimistically add to it
    if (!error) { // if context error is not set after creation attempt
      setNewWorkspaceName('');
      fetchWorkspaces(); // Refresh list
    }
  };

  return (
    <div className="workspace-selection-container">
      <div className="workspace-selection-box">
        <h1>Select or Create Workspace</h1>

        {isLoading && <p>Loading workspace...</p>}
        {error && <p className="error-text">{error}</p>}

        <div className="workspace-list-section">
          <h2>Available Workspaces</h2>
          {isLoadingList && <p>Loading list...</p>}
          {listError && <p className="error-text">{listError}</p>}
          {workspaces.length === 0 && !isLoadingList && <p>No workspaces found. Create one below.</p>}
          <ul>
            {workspaces.map((ws) => (
              <li key={ws.name}>
                <button onClick={() => handleSelectWorkspace(ws.name)} className="workspace-button">
                  {ws.name} <span className="workspace-path">({ws.env.CWD || ws.path})</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="workspace-create-section">
          <h2>Create New Workspace</h2>
          <form onSubmit={handleCreateWorkspace}>
            <input
              type="text"
              value={newWorkspaceName}
              onChange={(e) => setNewWorkspaceName(e.target.value)}
              placeholder="Enter new workspace name"
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !newWorkspaceName.trim()} className="button-primary">
              {isLoading ? 'Creating...' : 'Create and Open'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default WorkspaceSelectionView;