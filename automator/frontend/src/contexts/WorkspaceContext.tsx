import React, { createContext, useState, useContext, ReactNode, useEffect } from 'react';
import { Workspace } from '../types';
import * as api from '../services/api';

interface WorkspaceContextType {
  currentWorkspace: Workspace | null;
  selectWorkspace: (workspaceName: string) => Promise<void>;
  createAndSelectWorkspace: (workspaceName: string) => Promise<void>;
  clearWorkspace: () => void;
  isLoading: boolean;
  error: string | null;
}

const WorkspaceContext = createContext<WorkspaceContextType | undefined>(undefined);

export const WorkspaceProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Attempt to load workspace from local storage on initial load
  useEffect(() => {
    const storedWorkspaceName = localStorage.getItem('selectedWorkspaceName');
    if (storedWorkspaceName) {
      selectWorkspace(storedWorkspaceName);
    }
  }, []);

  const selectWorkspace = async (workspaceName: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const workspaceDetails = await api.getWorkspaceDetails(workspaceName);
      setCurrentWorkspace(workspaceDetails);
      localStorage.setItem('selectedWorkspaceName', workspaceName);
    } catch (err: any) {
      setError(err.message || 'Failed to select workspace');
      setCurrentWorkspace(null);
      localStorage.removeItem('selectedWorkspaceName');
    }
    setIsLoading(false);
  };

  const createAndSelectWorkspace = async (workspaceName: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const newWorkspace = await api.createWorkspace({ name: workspaceName });
      setCurrentWorkspace(newWorkspace);
      localStorage.setItem('selectedWorkspaceName', newWorkspace.name);
    } catch (err: any) {
      setError(err.message || 'Failed to create workspace');
      setCurrentWorkspace(null);
      localStorage.removeItem('selectedWorkspaceName');
    }
    setIsLoading(false);
  };

  const clearWorkspace = () => {
    setCurrentWorkspace(null);
    localStorage.removeItem('selectedWorkspaceName');
  };

  return (
    <WorkspaceContext.Provider value={{
      currentWorkspace,
      selectWorkspace,
      createAndSelectWorkspace,
      clearWorkspace,
      isLoading,
      error
    }}>
      {children}
    </WorkspaceContext.Provider>
  );
};

export const useWorkspace = (): WorkspaceContextType => {
  const context = useContext(WorkspaceContext);
  if (context === undefined) {
    throw new Error('useWorkspace must be used within a WorkspaceProvider');
  }
  return context;
};