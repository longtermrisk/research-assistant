import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import AppLayout from '../../components/AppLayout/AppLayout';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import {
  ApiChatMessage, ThreadDetail, Agent,
  ContentBlock, MessageRole, TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock
} from '../../types';
import './MainView.css';
import { getMessagesSSE } from '../../services/api';

// Function to recursively extract text from content blocks for copying
const extractTextForCopy = (block: ContentBlock, messagesForContext: ApiChatMessage[], currentWorkspaceName?: string): string => {
  switch (block.type) {
    case 'text':
      return (block as TextBlock).text;
    case 'image':
      return "[Image Content]"; // Placeholder for images
    case 'tool_use':
      const tuBlock = block as ToolUseBlock;
      let toolUseText = `Tool Call: ${tuBlock.name} (ID: ${tuBlock.id})\nInput:\n${JSON.stringify(tuBlock.input, null, 2)}`;
      // Find corresponding tool_result for this tool_use to append its content
      let correspondingResultText = "";
      // Search within the same message first
      // (This part of logic might be complex depending on how tool results are structured relative to uses)
      // For now, we'll rely on the combined rendering for full context.
      // A more sophisticated copy would need to traverse messages to find the result.
      return toolUseText;
    case 'tool_result':
      const trBlock = block as ToolResultBlock;
      let resultText = `Result for Tool Call ID: ${trBlock.tool_use_id}\n`;
      if (trBlock.meta?.thread_id && currentWorkspaceName) {
        resultText += `(Subagent Thread: /workspace/${currentWorkspaceName}/thread/${trBlock.meta.thread_id}`
        if (typeof trBlock.meta.message_start === 'number') {
          resultText += `#message-${trBlock.meta.message_start}`
        }
        resultText += ")\n";
      }
      resultText += trBlock.content.map(cb => extractTextForCopy(cb, messagesForContext, currentWorkspaceName)).join('\n');
      return resultText;
    default:
      return "[Unsupported Block Type]";
  }
};


const RenderContentBlock: React.FC<{
  block: ContentBlock;
  index: number;
  allBlocks: ContentBlock[];
  messages: ApiChatMessage[];
  workspaceName?: string;
}> = ({ block, index, allBlocks, messages, workspaceName }) => {
  // Helper to find the corresponding tool_use block for a tool_result block
  const findCorrespondingToolUse = (toolResultId: string): ToolUseBlock | undefined => {
    for (const msg of messages) {
      for (const b of msg.content) {
        if (b.type === 'tool_use' && b.id === toolResultId) {
          return b as ToolUseBlock;
        }
      }
    }
    return undefined;
  };

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
      const correspondingToolUse = findCorrespondingToolUse(toolResultBlock.tool_use_id);
      let resultPrefix = `Result for Tool Call ID: ${toolResultBlock.tool_use_id}`;
      if (correspondingToolUse) {
        resultPrefix = `Result for ${correspondingToolUse.name} (ID: ${toolResultBlock.tool_use_id})`;
      }

      // Subagent link rendering
      let subagentLinkElement: React.ReactNode = null;
      if (toolResultBlock.meta?.thread_id && workspaceName) {
        const subThreadPath = `/workspace/${workspaceName}/thread/${toolResultBlock.meta.thread_id}`;
        const linkTarget = typeof toolResultBlock.meta.message_start === 'number' 
          ? `${subThreadPath}#message-${toolResultBlock.meta.message_start}`
          : subThreadPath;
        subagentLinkElement = (
          <div className="subagent-link">
            <Link to={linkTarget}>
              View Subagent Thread: {toolResultBlock.meta.thread_id}
              {typeof toolResultBlock.meta.message_start === 'number' && ` (from message ${toolResultBlock.meta.message_start})`}
            </Link>
          </div>
        );
      }

      return (
        <div key={index} className="tool-result-block">
          <strong>{resultPrefix}</strong>
          {subagentLinkElement}
          {(toolResultBlock.content).map((contentItem, idx) => (
            <RenderContentBlock 
              key={idx} 
              block={contentItem} 
              index={idx} 
              allBlocks={toolResultBlock.content} 
              messages={messages} 
              workspaceName={workspaceName}
            />
          ))}
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
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);


  useEffect(() => {
    if (currentWorkspace && currentWorkspace.name) {
      api.listAgents(currentWorkspace.name)
        .then(setAvailableAgents)
        .catch(err => {
          console.error('Failed to load agents:', err);
          setError('Failed to load agents: ' + err.message);
        });
    }
  }, [currentWorkspace]);

  useEffect(() => {
    if (!currentWorkspace || !workspaceName) return;

    if (threadId) {
      setIsLoading(true);
      setError(null);
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
        setMessages(prevMessages => {
          // Merge logic for consecutive assistant messages if the last user message was tool-only
          const lastUserMessage = prevMessages.slice().reverse().find(m => m.role === MessageRole.User);
          if (lastUserMessage && lastUserMessage.content.every(c => c.type === 'tool_result')) {
            const lastAssistantMessage = prevMessages.slice().reverse().find(m => m.role === MessageRole.Assistant);
            if (lastAssistantMessage && newMessageData.role === MessageRole.Assistant) {
              // This is a new assistant message, and the one before it (after user tool results) was also assistant.
              // This scenario might indicate a continuation rather than direct merging of content blocks.
              // For now, simple append. More complex merging (e.g. consecutive text blocks) could be added.
            }
          }
          return [...prevMessages, newMessageData];
        });
      };
      sse.onerror = (err) => {
        console.error('SSE Error:', err);
        setError('Connection error with server updates.');
        sse.close();
      };
      return () => sse.close();
    } else {
      setCurrentThread(null);
      setMessages([]);
      const agentIdFromLocation = (location.state as { agentId?: string })?.agentId;
      if (agentIdFromLocation) {
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
    // Navigate to the base workspace/threadId path to clear any specific agent selection from state
    // The message input area will then appear for the selected agent.
    navigate(`/workspace/${workspaceName}`); 
  }

  const handleCopyMessage = (msg: ApiChatMessage, msgIndex: number) => {
    let contentToCopy = "";
    if (msg.role === MessageRole.Assistant) {
      // For assistant messages, construct a string that includes tool calls and their results
      // This requires iterating through its content blocks and potentially the next message if it contains results
      msg.content.forEach(block => {
        contentToCopy += extractTextForCopy(block, messages, workspaceName) + "\n";
        if (block.type === 'tool_use') {
          // Attempt to find and append the corresponding tool_result from the *next* message if it's a user message full of tool_results
          const nextMessage = messages[msgIndex + 1];
          if (nextMessage && nextMessage.role === MessageRole.User && nextMessage.content.every(c => c.type === 'tool_result')) {
            const resultBlock = nextMessage.content.find(b => (b as ToolResultBlock).tool_use_id === block.id) as ToolResultBlock | undefined;
            if (resultBlock) {
              contentToCopy += extractTextForCopy(resultBlock, messages, workspaceName) + "\n";
            }
          } else if (msg.content.some(b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === block.id)){
            // If tool_result is within the same assistant message
            const resultBlock = msg.content.find(b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === block.id) as ToolResultBlock | undefined;
            if (resultBlock) {
                 // contentToCopy += extractTextForCopy(resultBlock, messages, workspaceName) + "\n"; // Already handled by the iteration
            }
          }
        }
      });
    } else {
      // For user or system messages, just join the text from text blocks
      contentToCopy = msg.content
        .filter(block => block.type === 'text')
        .map(block => (block as TextBlock).text)
        .join('\n');
    }
    navigator.clipboard.writeText(contentToCopy.trim())
      .then(() => {
        setCopiedMessageIndex(msgIndex);
        setTimeout(() => setCopiedMessageIndex(null), 2000); // Hide "Copied!" after 2s
      })
      .catch(err => console.error("Failed to copy: ", err));
  };


  if (!currentWorkspace) {
    return <AppLayout><div>Loading workspace data or workspace not selected...</div></AppLayout>;
  }
  
  if (!threadId && !selectedAgentForNewThread) {
    return (
      <AppLayout>
        <div className="new-thread-agent-selection">
          <h2>Select an Agent to Start a New Thread</h2>
          {error && <p className="error-text">Error: {error}</p>}
          {availableAgents.length === 0 && !isLoading && (
            <p>No agents available. <Link to={`/workspace/${workspaceName}/agents`}>Create one?</Link></p>
          )}
          {isLoading && <p>Loading agents...</p>}
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

  // Prepare messages for rendering, filtering out user messages that only contain tool results
  const processedMessages = messages.filter(msg => {
    if (msg.role === MessageRole.User && msg.content.length > 0 && msg.content.every(block => block.type === 'tool_result')) {
      return false; // Don't render user messages that are only tool results
    }
    return true;
  });


  return (
    <AppLayout>
      <div className="chat-view-container">
        {isLoading && !processedMessages.length && <div className="loading-chat">Loading messages...</div>}
        {error && <div className="error-message">Error: {error}</div>}
        
        <div className="messages-area" ref={chatContainerRef} onScroll={handleScroll}>
          {processedMessages.map((msg, msgIndex) => {
            // Find the original index in the `messages` array for accurate copy context
            const originalMsgIndex = messages.findIndex(originalMsg => originalMsg === msg);
            return (
            <div key={originalMsgIndex} className={`message-bubble ${msg.role}`}>
              <div className="message-header">
                <span className="role">{msg.role.toUpperCase()}</span>
                <button onClick={() => handleCopyMessage(msg, originalMsgIndex)} className="copy-button">
                  {copiedMessageIndex === originalMsgIndex ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <div className="message-content">
                {msg.content.map((block, index) => {
                  // Logic for merging assistant message content if previous user message was tool-only
                  // This is complex and needs careful state management.
                  // The current filtering of `processedMessages` handles hiding the tool-only user message.
                  // The `renderToolCycle` logic handles displaying tool_use and tool_result together.

                  // If it's a tool_use, find its corresponding tool_result (potentially in the next original message)
                  if (block.type === 'tool_use') {
                    const toolUse = block as ToolUseBlock;
                    let toolResult: ToolResultBlock | undefined = undefined;

                    // Look in the same message first (some models might return it this way)
                    const resultInSameMessage = msg.content.find(
                      b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === toolUse.id
                    ) as ToolResultBlock | undefined;

                    if (resultInSameMessage) {
                      toolResult = resultInSameMessage;
                    } else {
                      // Look in the *next* message in the *original* messages array, if it's a user message full of tool_results
                      const nextOriginalMessage = messages[originalMsgIndex + 1];
                      if (nextOriginalMessage && nextOriginalMessage.role === MessageRole.User && 
                          nextOriginalMessage.content.every(c => c.type === 'tool_result')) {
                        toolResult = nextOriginalMessage.content.find(
                          b => (b as ToolResultBlock).tool_use_id === toolUse.id
                        ) as ToolResultBlock | undefined;
                      }
                    }
                    
                    return (
                      <div key={index} className="tool-cycle-block">
                        <RenderContentBlock block={toolUse} index={index} allBlocks={msg.content} messages={messages} workspaceName={workspaceName}/>
                        {toolResult && <RenderContentBlock block={toolResult} index={-1} allBlocks={[]} messages={messages} workspaceName={workspaceName} />}
                      </div>
                    );
                  }
                  // If it's a tool_result, it should have been handled by its corresponding tool_use.
                  // If it's rendered here, it means it wasn't paired (e.g. result in user message without preceding assistant tool_use)
                  // or it's part of a tool_result's own content (e.g. text within a tool_result).
                  if (block.type === 'tool_result') {
                     // Check if this tool_result was already rendered as part of a tool_cycle
                    let wasHandled = false;
                    // Check previous blocks in the same message
                    for(let i = 0; i < index; i++) {
                        const prevBlock = msg.content[i] as ContentBlock;
                        if(prevBlock.type === 'tool_use' && (prevBlock as ToolUseBlock).id === (block as ToolResultBlock).tool_use_id) {
                            wasHandled = true;
                            break;
                        }
                    }
                    // Check if it was handled by a tool_use in the *previous* original message
                    // This covers the case where assistant makes a tool_use, and user replies with tool_result
                    if (!wasHandled && originalMsgIndex > 0) {
                        const prevOriginalMessage = messages[originalMsgIndex - 1];
                        if (prevOriginalMessage.role === MessageRole.Assistant && 
                            prevOriginalMessage.content.some(b => b.type === 'tool_use' && (b as ToolUseBlock).id === (block as ToolResultBlock).tool_use_id)) {
                            wasHandled = true;
                        }
                    }
                    if (wasHandled) return null; 
                  }
                  return <RenderContentBlock block={block} index={index} allBlocks={msg.content} messages={messages} workspaceName={workspaceName} />;
                })}
              </div>
            </div>
          )})}
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