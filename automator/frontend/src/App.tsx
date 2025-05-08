import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { WorkspaceProvider, useWorkspace } from './contexts/WorkspaceContext';
import WorkspaceSelectionView from './views/WorkspaceSelectionView/WorkspaceSelectionView';
import MainView from './views/MainView/MainView';
import AgentManagementView from './views/AgentManagementView/AgentManagementView';

import './styles/global.css';
import './styles/theme.css';

const AppContent: React.FC = () => {
  const { currentWorkspace, isLoading } = useWorkspace();

  if (isLoading) {
    return <div style={{ textAlign: 'center', marginTop: '50px', fontSize: '1.2em', color: 'var(--foreground-primary)' }}>Loading workspace data...</div>;
  }

  return (
    <Routes>
      <Route 
        path="/"
        element={
          currentWorkspace ? (
            // Navigate to the dynamic path using the actual current workspace name
            <Navigate to={`/workspace/${currentWorkspace.name}`} />
          ) : (
            <WorkspaceSelectionView />
          )
        }
      />
      
      {/* Routes accessible only when a workspace is selected */}
      {/* Use :workspaceName in the path for useParams to correctly capture it */}
      <Route path="/workspace/:workspaceName">
        <Route index element={<MainView />} /> {/* Matches /workspace/:workspaceName */} 
        <Route path="thread/:threadId" element={<MainView />} /> {/* Matches /workspace/:workspaceName/thread/:threadId */}
        <Route path="agents" element={<AgentManagementView />} /> {/* Matches /workspace/:workspaceName/agents */}
      </Route>

      {/* Fallback redirect if trying to access a workspace-specific path without a selected workspace */}
      {!currentWorkspace && (
        // This catches any attempt to go to /workspace/* if context is lost or not set
        <Route path="/workspace/*" element={<Navigate to="/" replace />} />
      )}

      {/* Catch-all for any other undefined paths, redirect to workspace selection or main view */}
      <Route 
        path="*" 
        element={
          currentWorkspace ? (
            <Navigate to={`/workspace/${currentWorkspace.name}`} replace /> 
          ) : (
            <Navigate to="/" replace />
          )
        } 
      />
    </Routes>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <WorkspaceProvider>
        <AppContent />
      </WorkspaceProvider>
    </Router>
  );
};

export default App;