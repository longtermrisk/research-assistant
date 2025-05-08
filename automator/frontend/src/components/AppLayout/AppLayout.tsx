import React, { useState, useEffect } from 'react';
import { Link, useParams, useNavigate, useLocation } from 'react-router-dom';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import { ThreadSummary } from '../../types';
import './AppLayout.css';

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const { currentWorkspace, clearWorkspace } = useWorkspace();
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const navigate = useNavigate();
  const params = useParams<{ workspaceName?: string; threadId?: string }>();
  const location = useLocation(); // To check current path for active thread

  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [isLoadingThreads, setIsLoadingThreads] = useState<boolean>(false);

  const workspaceNameForLinks = currentWorkspace?.name || params.workspaceName;

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

  const handleLogout = () => {
    clearWorkspace();
    navigate('/');
  };

  if (!workspaceNameForLinks) {
    console.warn('AppLayout rendered without a valid workspace name. Redirecting.');
    navigate('/'); 
    return <div className="app-layout-error">Preparing workspace...</div>;
  }

  return (
    <div className={`app-layout ${isSidebarExpanded ? 'sidebar-expanded' : 'sidebar-collapsed'}`}>
      <header className="top-bar">
        <div className="top-bar-left">
          <button onClick={() => setIsSidebarExpanded(!isSidebarExpanded)} className="sidebar-toggle-btn">
            {isSidebarExpanded ? '<' : '>'} 
          </button>
          <span className="workspace-name">{currentWorkspace?.name || workspaceNameForLinks || 'Workspace'}</span>
        </div>
        <div className="top-bar-right">
          <Link to={`/workspace/${workspaceNameForLinks}/agents`} className="nav-button">Agents</Link>
          <button onClick={handleLogout} className="nav-button logout-button">Switch Workspace</button>
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
                title={thread.id} // Show full ID on hover
              >
                {/* Task: show the first words of the first user message. Needs backend change or fetch details.
                    For now, just showing thread ID or a placeholder. */}
                Thread {thread.id.substring(0, 8)}...
              </Link>
            ))}
          </nav>
        </div>
      </aside>

      <main className="main-content">
        {currentWorkspace ? children : <div className="app-layout-info">Loading workspace content...</div>}
      </main>
    </div>
  );
};

export default AppLayout;