import React, { useState, useEffect } from 'react';
import { Link, useParams, useNavigate, useLocation } from 'react-router-dom';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import CommandPalette from '../CommandPalette/CommandPalette';
import * as api from '../../services/api';
import { ThreadSummary } from '../../types';
import './AppLayout.css';

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const { currentWorkspace } = useWorkspace();
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const navigate = useNavigate();
  const params = useParams<{ workspaceName?: string; threadId?: string }>();
  const location = useLocation(); // To check current path for active thread

  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState<boolean>(false);

  const workspaceNameForLinks = currentWorkspace?.name || params.workspaceName;

  // Add keyboard shortcut for command palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'P') {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (currentWorkspace && currentWorkspace.name) {
      setIsLoadingThreads(true);
      api.listThreads(currentWorkspace.name)
        .then(fetchedThreads => {
          // Sort threads by some logic if needed, e.g., last modified (not available in summary)
          // For now, default order or sort by ID if desired.
          setThreads(fetchedThreads);
        })
        .catch(err => {
          console.error('Failed to load threads for sidebar:', err);
          // Optionally set an error state for threads
        })
        .finally(() => {
          setIsLoadingThreads(false);
        });
    }
  }, [currentWorkspace, location]); // Re-fetch if workspace changes or location changes (e.g. new thread created and navigated to)

  const handleWorkspaceSwitch = () => {
    setIsCommandPaletteOpen(true);
  };

  if (!workspaceNameForLinks) {
    console.warn('AppLayout rendered without a valid workspace name. Redirecting.');
    navigate('/'); 
    return <div className="app-layout-error">Preparing workspace...</div>;
  }

  return (
    <div className="app-layout">
      <header className="top-bar">
        <div className="top-bar-left">
          <button onClick={handleWorkspaceSwitch} className="workspace-button" title="Switch workspace (Cmd+Shift+P)">
            <span className="workspace-name">{currentWorkspace?.name || workspaceNameForLinks || 'Workspace'}</span>
            <span className="workspace-path">{currentWorkspace?.env.CWD || currentWorkspace?.path || ''}</span>
          </button>
        </div>
        <div className="top-bar-right">
          <Link to={`/workspace/${workspaceNameForLinks}/agents`} className="nav-button">Agents</Link>
        </div>
      </header>

      <aside className="left-sidebar">
        <div className="sidebar-content">
          <Link to={`/workspace/${workspaceNameForLinks}`} className="new-thread-button">+ New Thread</Link>
          
          <h3 className="sidebar-heading">Threads</h3>
          {isLoadingThreads && <p className="loading-text">Loading threads...</p>}
          <nav className="thread-list">
            {threads.length === 0 && !isLoadingThreads && (
              <p className="no-threads-text">No threads yet.</p>
            )}
            {threads.map(thread => (
              <Link 
                key={thread.id} 
                to={`/workspace/${workspaceNameForLinks}/thread/${thread.id}`}
                className={`thread-item ${params.threadId === thread.id ? 'active' : ''}`}
                title={thread.first_user_message_preview || thread.id} // Show full preview or ID on hover
              >
                {thread.first_user_message_preview || `Thread ${thread.id.substring(0, 8)}...`}
              </Link>
            ))}
          </nav>
        </div>
      </aside>

      <main className="main-content">
        {currentWorkspace ? children : <div className="app-layout-info">Loading workspace content...</div>}
      </main>

      <CommandPalette 
        isOpen={isCommandPaletteOpen} 
        onClose={() => setIsCommandPaletteOpen(false)} 
      />
    </div>
  );
};

export default AppLayout;