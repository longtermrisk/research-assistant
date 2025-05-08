import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom'; // Added Link here
import AppLayout from '../../components/AppLayout/AppLayout';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import {
  ApiChatMessage, ThreadDetail, Agent, ThreadSummary,
  ContentBlock, MessageRole, TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock
} from '../../types';
import './MainView.css';
import { getMessagesSSE } from '../../services/api';

// Placeholder for actual content block renderers
const renderContentBlock = (block: ContentBlock, index: number, allBlocks: ContentBlock[], messages: ApiChatMessage[]) => {
  let correspondingToolUse: ToolUseBlock | undefined;
  if (block.type === 'tool_result') {
    for (const msg of messages) {
      for (const b of msg.content) {
        if (b.type === 'tool_use' && b.id === (block as ToolResultBlock).tool_use_id) {
          correspondingToolUse = b as ToolUseBlock;
          break;
        }
      }
      if (correspondingToolUse) break;
    }
  }

  switch (block.type) {
    case 'text':
      return <div key={index} className="text-block">{(block as TextBlock).text}</div>;
    case 'image':
      const imageBlock = block as ImageBlock;
      return <div key={index} className="image-block"><img src={`data:${imageBlock.source.media_type};base64,${imageBlock.source.data}`} alt="Uploaded content" /></div>;
    case 'tool_use':
      const toolUseBlock = block as ToolUseBlock;
      return (
        <div key={index} className="tool-use-block">
          <strong>Tool Call: {toolUseBlock.name} (ID: {toolUseBlock.id})</strong>
          <pre>{JSON.stringify(toolUseBlock.input, null, 2)}</pre>
        </div>
      );
    case 'tool_result':
      const toolResultBlock = block as ToolResultBlock;
      let resultPrefix = `Result for Tool Call ID: ${toolResultBlock.tool_use_id}`;
      if (correspondingToolUse) {
         resultPrefix = `Result for ${correspondingToolUse.name} (ID: ${toolResultBlock.tool_use_id})`;
      }
      return (
        <div key={index} className="tool-result-block">
          <strong>{resultPrefix}</strong>
          {(toolResultBlock.content).map((contentItem, idx) => renderContentBlock(contentItem, idx, toolResultBlock.content, messages))}
        </div>
      );
    default:
      return <div key={index} className="unknown-block">Unsupported block type</div>;
  }
};

const MainView: React.FC = () => {
  const { currentWorkspace } = useWorkspace();
  const { workspaceName, threadId } = useParams<{ workspaceName: string; threadId?: string; }>();
  const navigate = useNavigate();
  const location = useLocation();

  const [currentThread, setCurrentThread] = useState<ThreadDetail | null>(null);
  const [messages, setMessages] = useState<ApiChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [selectedAgentForNewThread, setSelectedAgentForNewThread] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  const [isScrolledUp, setIsScrolledUp] = useState(false);

  useEffect(() => {
    console.log('[MainView Original] Mounted. Workspace:', currentWorkspace?.name, 'Params:', workspaceName, threadId);
    if (currentWorkspace && currentWorkspace.name) {
      api.listAgents(currentWorkspace.name)
        .then(setAvailableAgents)
        .catch(err => {
          console.error('Failed to load agents:', err);
          setError('Failed to load agents: ' + err.message);
        });
    } else {
      console.warn('[MainView Original] currentWorkspace or its name is not defined on mount for listing agents.');
    }
  }, [currentWorkspace]); // currentWorkspace.name removed as dependency, currentWorkspace itself is enough

  useEffect(() => {
    if (!currentWorkspace || !workspaceName) {
      console.log('[MainView Original] SSE/ThreadDetails effect: Aborting, no currentWorkspace or workspaceName.');
      return;
    }

    if (threadId) {
      console.log(`[MainView Original] SSE/ThreadDetails effect: Fetching details for thread ${threadId}`);
      setIsLoading(true);
      setError(null); // Clear previous errors
      api.getThreadDetails(workspaceName, threadId)
        .then(threadDetails => {
          setCurrentThread(threadDetails);
          setMessages(threadDetails.messages || []);
          setIsLoading(false);
        })
        .catch(err => {
          console.error('Failed to load thread details:', err);
          setError(err.message || 'Failed to load thread');
          setIsLoading(false);
        });

      const sse = getMessagesSSE(workspaceName, threadId);
      sse.onmessage = (event) => {
        const newMessageData = JSON.parse(event.data) as ApiChatMessage;
        setMessages(prevMessages => [...prevMessages, newMessageData]);
      };
      sse.onerror = (err) => {
        console.error('SSE Error:', err);
        setError('Connection error with server updates.');
        sse.close();
      };
      return () => sse.close();
    } else {
      console.log('[MainView Original] SSE/ThreadDetails effect: No threadId. Setting up for new thread.');
      setCurrentThread(null);
      setMessages([]);
      const agentIdFromLocation = (location.state as { agentId?: string })?.agentId;
      if(agentIdFromLocation) {
        setSelectedAgentForNewThread(agentIdFromLocation);
      } else {
        setSelectedAgentForNewThread(null);
      }
    }
  }, [currentWorkspace, workspaceName, threadId, location.state]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      setIsScrolledUp(scrollTop < scrollHeight - clientHeight - 50);
    }
  };

  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    chatContainer?.addEventListener('scroll', handleScroll);
    return () => {
      chatContainer?.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return;
    if (!currentWorkspace || !workspaceName) return;

    setError(null);
    setIsLoading(true);

    if (currentThread && threadId) {
      try {
        await api.postMessage(workspaceName, threadId, { query: newMessage });
        setNewMessage('');
      } catch (err: any) {
        setError(err.message || 'Failed to send message');
      }
    } else if (selectedAgentForNewThread) {
      try {
        const newThreadSummary = await api.createThread(workspaceName, {
          agent_id: selectedAgentForNewThread,
          initial_query: newMessage,
        });
        setNewMessage('');
        setSelectedAgentForNewThread(null);
        navigate(`/workspace/${workspaceName}/thread/${newThreadSummary.id}`);
      } catch (err: any) {
        setError(err.message || 'Failed to create thread');
      }
    }
    setIsLoading(false);
  };

  const handleAgentSelectionForNewThread = (agentId: string) => {
    setSelectedAgentForNewThread(agentId);
    navigate(`/workspace/${workspaceName}`); 
  }

  if (!currentWorkspace) {
    // This might be hit if the context is somehow lost, or on initial load before context is ready.
    // App.tsx routing should prevent this view from rendering if currentWorkspace is truly null.
    return <AppLayout><div>Loading workspace data or workspace not selected...</div></AppLayout>;
  }
  
  if (!threadId && !selectedAgentForNewThread) {
    return (
      <AppLayout>
        <div className="new-thread-agent-selection">
          <h2>Select an Agent to Start a New Thread</h2>
          {error && <p className="error-text">Error: {error}</p>} {/* Display errors here */}
          {availableAgents.length === 0 && !isLoading && (
            <p>No agents available. <Link to={`/workspace/${workspaceName}/agents`}>Create one?</Link></p>
          )}
          {isLoading && <p>Loading agents...</p>} {/* Loading state for agents */}
          <div className="agent-list">
            {availableAgents.map(agent => (
              <button key={agent.id} onClick={() => handleAgentSelectionForNewThread(agent.id)} className="agent-select-button">
                {agent.id} ({agent.model})
              </button>
            ))}
          </div>
        </div>
      </AppLayout>
    );
  }
  
  const showChatInterface = threadId || selectedAgentForNewThread;

  return (
    <AppLayout>
      <div className="chat-view-container">
        {isLoading && !messages.length && <div className="loading-chat">Loading messages...</div>}
        {error && <div className="error-message">Error: {error}</div>}
        
        <div className="messages-area" ref={chatContainerRef} onScroll={handleScroll}>
          {messages.map((msg, msgIndex) => (
            <div key={msgIndex} className={`message-bubble ${msg.role}`}>
              <div className="message-header">
                <span className="role">{msg.role.toUpperCase()}</span>
              </div>
              <div className="message-content">
                {msg.content.map((block, index) => {
                  if (block.type === 'tool_use') {
                    const toolUse = block as ToolUseBlock;
                    let toolResult: ToolResultBlock | undefined = undefined;
                    if (msg.role === MessageRole.Assistant && (msgIndex + 1) < messages.length) {
                        const nextMsg = messages[msgIndex + 1];
                        if (nextMsg.role === MessageRole.User && nextMsg.content.every(b => b.type === 'tool_result')) {
                            toolResult = nextMsg.content.find(b => (b as ToolResultBlock).tool_use_id === toolUse.id) as ToolResultBlock;
                        }
                    }
                    if (!toolResult) {
                        for (let i = index + 1; i < msg.content.length; i++) {
                            const nextBlock = msg.content[i];
                            if (nextBlock.type === 'tool_result' && (nextBlock as ToolResultBlock).tool_use_id === toolUse.id) {
                                toolResult = nextBlock as ToolResultBlock;
                                break;
                            }
                        }
                    }
                    return (
                      <div key={index} className="tool-cycle-block">
                        {renderContentBlock(toolUse, index, msg.content, messages)}
                        {toolResult && renderContentBlock(toolResult, -1, [], messages)}
                      </div>
                    );
                  }
                  if (block.type === 'tool_result') {
                    let wasHandled = false;
                    for(let i = 0; i < index; i++) {
                        const prevBlock = msg.content[i] as ContentBlock;
                        if(prevBlock.type === 'tool_use' && (prevBlock as ToolUseBlock).id === (block as ToolResultBlock).tool_use_id) {
                            wasHandled = true;
                            break;
                        }
                    }
                    if (!wasHandled && msg.role === MessageRole.User && msgIndex > 0) {
                        const prevMsg = messages[msgIndex -1];
                        if (prevMsg.role === MessageRole.Assistant) {
                            if(prevMsg.content.some(b => b.type === 'tool_use' && (b as ToolUseBlock).id === (block as ToolResultBlock).tool_use_id)) {
                                wasHandled = true;
                            }
                        }
                    }
                    if (wasHandled) return null; 
                  }
                  return renderContentBlock(block, index, msg.content, messages);
                })}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {isScrolledUp && (
          <div className="scroll-buttons">
            <button onClick={() => chatContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })}>Scroll to Top</button>
            <button onClick={scrollToBottom}>Scroll to Bottom</button>
          </div>
        )}

        {showChatInterface && (
          <div className="message-input-area">
            <textarea
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder={currentThread ? `Message ${currentThread.id}` : `Message to ${selectedAgentForNewThread || 'new thread'}...`}
              rows={1}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
            />
            <button onClick={handleSendMessage} disabled={isLoading || !newMessage.trim() || (!threadId && !selectedAgentForNewThread)}>
              {isLoading ? 'Sending...' : 'Send'}
            </button>
          </div>
        )}
      </div>
    </AppLayout>
  );
};

export default MainView;