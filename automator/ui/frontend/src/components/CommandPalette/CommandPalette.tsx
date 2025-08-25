import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import { Workspace } from '../../types';
import './CommandPalette.css';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

const CommandPalette: React.FC<CommandPaletteProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate();
  const { currentWorkspace, selectWorkspace, createAndSelectWorkspace } = useWorkspace();
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoadingWorkspaces, setIsLoadingWorkspaces] = useState(false);
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setSearchTerm('');
      setSelectedIndex(0);
      fetchWorkspaces();
      // Focus input when opened
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isOpen]);

  const fetchWorkspaces = async () => {
    setIsLoadingWorkspaces(true);
    try {
      const fetchedWorkspaces = await api.listWorkspaces();
      setWorkspaces(fetchedWorkspaces);
    } catch (err) {
      console.error('Failed to fetch workspaces:', err);
    }
    setIsLoadingWorkspaces(false);
  };

  const handleSelectWorkspace = async (workspaceName: string) => {
    if (workspaceName === currentWorkspace?.name) {
      onClose();
      return;
    }
    
    try {
      await selectWorkspace(workspaceName);
      // Navigate to the workspace root to show the "create thread" view
      navigate(`/workspace/${workspaceName}`);
      onClose();
    } catch (err) {
      console.error('Failed to select workspace:', err);
    }
  };

  const handleCreateWorkspace = async (workspaceName: string) => {
    if (!workspaceName.trim()) return;
    
    setIsCreatingWorkspace(true);
    try {
      await createAndSelectWorkspace(workspaceName.trim());
      // Navigate to the new workspace root to show the "create thread" view
      navigate(`/workspace/${workspaceName.trim()}`);
      onClose();
      // Refresh the workspace list
      fetchWorkspaces();
    } catch (err) {
      console.error('Failed to create workspace:', err);
    }
    setIsCreatingWorkspace(false);
  };

  const filteredWorkspaces = workspaces.filter(ws => 
    ws.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Check if search term is a potential new workspace name
  const isNewWorkspaceName = searchTerm.trim() && !workspaces.some(ws => 
    ws.name.toLowerCase() === searchTerm.toLowerCase()
  );

  // Create items array that includes both existing workspaces and the create option
  const allItems = [
    ...filteredWorkspaces.map(ws => ({ type: 'workspace' as const, workspace: ws })),
    ...(isNewWorkspaceName ? [{ type: 'create' as const, name: searchTerm.trim() }] : [])
  ];

  // Reset selected index when search changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchTerm]);

  // Keyboard event handling
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;
      
      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, allItems.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const selectedItem = allItems[selectedIndex];
        if (selectedItem) {
          if (selectedItem.type === 'workspace') {
            handleSelectWorkspace(selectedItem.workspace.name);
          } else if (selectedItem.type === 'create') {
            handleCreateWorkspace(selectedItem.name);
          }
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, allItems, handleSelectWorkspace, handleCreateWorkspace, onClose]);

  if (!isOpen) return null;

  return (
    <div className="command-palette-backdrop" onClick={onClose}>
      <div className="command-palette" onClick={e => e.stopPropagation()}>
        <div className="command-palette-header">
          <input
            ref={inputRef}
            type="text"
            placeholder="Switch workspace or type to create new..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="command-palette-input"
          />
        </div>
        
        <div className="command-palette-content">
          {isLoadingWorkspaces ? (
            <div className="command-palette-loading">Loading workspaces...</div>
          ) : allItems.length === 0 ? (
            <div className="command-palette-empty">
              {searchTerm ? 'No matching workspaces found' : 'No workspaces available'}
            </div>
          ) : (
            <div className="command-palette-list">
              {allItems.map((item, index) => {
                if (item.type === 'workspace') {
                  const workspace = item.workspace;
                  return (
                    <div
                      key={workspace.name}
                      className={`command-palette-item ${index === selectedIndex ? 'selected' : ''} ${workspace.name === currentWorkspace?.name ? 'current' : ''}`}
                      onClick={() => handleSelectWorkspace(workspace.name)}
                    >
                      <div className="workspace-info">
                        <div className="workspace-name">{workspace.name}</div>
                        <div className="workspace-path">{workspace.env.CWD || workspace.path}</div>
                      </div>
                      {workspace.name === currentWorkspace?.name && (
                        <div className="current-indicator">Current</div>
                      )}
                    </div>
                  );
                } else {
                  // Create workspace option
                  return (
                    <div
                      key={`create-${item.name}`}
                      className={`command-palette-item create-item ${index === selectedIndex ? 'selected' : ''} ${isCreatingWorkspace ? 'creating' : ''}`}
                      onClick={() => handleCreateWorkspace(item.name)}
                    >
                      <div className="workspace-info">
                        <div className="workspace-name">
                          <span className="create-icon">+ </span>
                          Create "{item.name}"
                        </div>
                        <div className="workspace-path">Create new workspace</div>
                      </div>
                      {isCreatingWorkspace && (
                        <div className="current-indicator">Creating...</div>
                      )}
                    </div>
                  );
                }
              })}
            </div>
          )}
        </div>
        
        <div className="command-palette-footer">
          <span className="command-palette-help">
            Use ↑↓ to navigate, Enter to select/create, Esc to close
          </span>
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;