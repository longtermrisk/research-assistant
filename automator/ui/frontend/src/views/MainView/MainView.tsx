import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate, useLocation, Link } from 'react-router-dom';
import AppLayout from '../../components/AppLayout/AppLayout';
import { useWorkspace } from '../../contexts/WorkspaceContext';
import * as api from '../../services/api';
import {
  ApiChatMessage, ThreadDetail, Agent,
  ContentBlock, MessageRole, TextBlock, ToolUseBlock, ToolResultBlock, ImageBlock, Base64ImageSource, FileSystemItem
} from '../../types';
import './MainView.css';
import { getMessagesSSE } from '../../services/api';
import { MessageBubble } from './MessageBubble';
import MessageInputArea from './MessageInputArea';

interface AttachedImage {
  id: string;
  file: File;
  base64: string;
  media_type: string;
}

const extractTextForCopy = (
  block: ContentBlock,
  messagesForContext: ApiChatMessage[],
  currentWorkspaceName?: string
): string => {
  switch (block.type) {
    case 'text':
      if (block.meta && block.meta.hidden) return '';
      return (block as TextBlock).text;
    case 'image':
      return '[Image Content]';
    case 'tool_use':
      const tuBlock = block as ToolUseBlock;
      return `Tool Call: ${tuBlock.name} (ID: ${tuBlock.id})\nInput:\n${JSON.stringify(
        tuBlock.input,
        null,
        2
      )}`;
    case 'tool_result':
      const trBlock = block as ToolResultBlock;
      let resultText = `Result for Tool Call ID: ${trBlock.tool_use_id}\n`;
      if (trBlock.meta?.thread_id && currentWorkspaceName) {
        resultText += `(Subagent Thread: /workspace/${currentWorkspaceName}/thread/${trBlock.meta.thread_id}`;
        if (typeof trBlock.meta.message_start === 'number') {
          resultText += `#message-${trBlock.meta.message_start}`;
        }
        resultText += ')\n';
      }
      resultText += trBlock.content
        .map((cb) => extractTextForCopy(cb, messagesForContext, currentWorkspaceName))
        .join('\n');
      return resultText;
    default:
      return '[Unsupported Block Type]';
  }
};

const MainView: React.FC = () => {
  const { currentWorkspace } = useWorkspace();
  const { workspaceName, threadId } = useParams<{ workspaceName: string; threadId?: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  const [currentThread, setCurrentThread] = useState<ThreadDetail | null>(null);
  const [messages, setMessages] = useState<ApiChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState<string>('');
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [selectedAgentForNewThread, setSelectedAgentForNewThread] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [selectedFilePaths, setSelectedFilePaths] = useState<Set<string>>(new Set());
  const [workspaceFiles, setWorkspaceFiles] = useState<FileSystemItem[]>([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState<boolean>(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  const [isScrolledUp, setIsScrolledUp] = useState(false);
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);
  const [isInterrupting, setIsInterrupting] = useState<boolean>(false);
  const [isAgentProcessing, setIsAgentProcessing] = useState<boolean>(false);

  useEffect(() => {
    if (currentWorkspace && currentWorkspace.name && workspaceName) {
      setIsLoadingFiles(true);
      api
        .listWorkspaceFiles(workspaceName)
        .then((files) => {
          setWorkspaceFiles(files);
          setIsLoadingFiles(false);
        })
        .catch((err) => {
          console.error('Failed to load workspace files:', err);
          setError('Failed to load workspace files: ' + err.message);
          setIsLoadingFiles(false);
        });

      api
        .listAgents(workspaceName)
        .then(setAvailableAgents)
        .catch((err) => {
          console.error('Failed to load agents:', err);
          setError((prev) => prev || 'Failed to load agents: ' + err.message);
        });
    }
  }, [currentWorkspace, workspaceName]);

  useEffect(() => {
    if (!currentWorkspace || !workspaceName) return;
    if (threadId) {
      setIsLoading(true);
      setError(null);
      api
        .getThreadDetails(workspaceName, threadId)
        .then((threadDetails) => {
          setCurrentThread(threadDetails);
          setMessages(threadDetails.messages || []);
          
          // Check if the last message indicates the agent might still be processing
          const lastMessage = threadDetails.messages?.[threadDetails.messages.length - 1];
          if (lastMessage && lastMessage.role === MessageRole.assistant) {
            const hasToolUse = lastMessage.content.some(block => block.type === 'tool_use');
            if (hasToolUse) {
              setIsAgentProcessing(true); // Agent might still be processing tool results
            }
          }
          
          setIsLoading(false);
        })
        .catch((err) => {
          console.error('Failed to load thread details:', err);
          setError(err.message || 'Failed to load thread');
          setIsLoading(false);
        });

      const sse = getMessagesSSE(workspaceName, threadId);
      sse.onmessage = (event) => {
        const newMessageData = JSON.parse(event.data) as ApiChatMessage;
        
        // Check if this is an assistant message that indicates processing is complete
        if (newMessageData.role === MessageRole.assistant) {
          // If the assistant message has no tool_use blocks, the agent is likely done processing
          const hasToolUse = newMessageData.content.some(block => block.type === 'tool_use');
          console.log(`[SSE] Assistant message received, hasToolUse: ${hasToolUse}`);
          if (!hasToolUse) {
            console.log('[SSE] Agent processing complete (no tool_use blocks)');
            setIsAgentProcessing(false);
          } else {
            console.log('[SSE] Agent still processing (has tool_use blocks)');
            setIsAgentProcessing(true);
          }
        }
        
        // If we receive an error message, agent has stopped processing
        if (newMessageData.meta && newMessageData.meta.error) {
          console.log('[SSE] Error message received, stopping agent processing');
          setIsAgentProcessing(false);
        }
        
        setMessages((prevMessages) => [...prevMessages, newMessageData]);
      };
      sse.onerror = (err) => {
        console.error('SSE Error:', err);
        setError((prev) => prev || 'Connection error with server updates.');
        setIsAgentProcessing(false); // Stop processing on SSE error
        sse.close();
      };
      return () => {
        sse.close();
        setIsAgentProcessing(false); // Stop processing when SSE closes
      };
    } else {
      setCurrentThread(null);
      setMessages([]);
      const agentIdFromLocation = (location.state as { agentId?: string })?.agentId;
      setSelectedAgentForNewThread(agentIdFromLocation || null);
    }
  }, [currentWorkspace, workspaceName, threadId, location.state]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Auto-scroll to bottom only if user is not scrolled up
  useEffect(() => {
    if (!isScrolledUp) {
      scrollToBottom();
    }
  }, [messages, attachedImages, isScrolledUp]);

  const handleScroll = () => {
    if (chatContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = chatContainerRef.current;
      setIsScrolledUp(scrollTop < scrollHeight - clientHeight - 50);
    }
  };

  useEffect(() => {
    const chatContainer = chatContainerRef.current;
    chatContainer?.addEventListener('scroll', handleScroll);
    return () => chatContainer?.removeEventListener('scroll', handleScroll);
  }, []);

  const handleSendMessage = async () => {
    if (!newMessage.trim() && attachedImages.length === 0) return;
    if (!currentWorkspace || !workspaceName) return;

    setError(null);
    setIsLoading(true);

    const contentBlocks: ContentBlock[] = [];
    if (newMessage.trim()) {
      contentBlocks.push({ type: 'text', text: newMessage.trim() });
    }
    attachedImages.forEach((img) => {
      contentBlocks.push({
        type: 'image',
        source: {
          type: 'base64',
          media_type: img.media_type,
          data: img.base64,
        } as Base64ImageSource,
      });
    });

    const mentionedFilePaths = Array.from(selectedFilePaths);
        const payload = {
          content: contentBlocks,
          mentioned_file_paths: mentionedFilePaths.length > 0 ? mentionedFilePaths : undefined,
        };

        if (currentThread && threadId) {
          try {
            await api.postMessage(workspaceName, threadId, payload);
            setNewMessage('');
            setAttachedImages([]);
            setSelectedFilePaths(new Set());
            setIsAgentProcessing(true); // Agent will start processing
          } catch (err: any) {
            setError(err.message || 'Failed to send message');
          }
        } else if (selectedAgentForNewThread) {
          try {
            const newThreadSummary = await api.createThread(workspaceName, {
              agent_id: selectedAgentForNewThread,
              initial_content: contentBlocks,
              mentioned_file_paths: mentionedFilePaths.length > 0 ? mentionedFilePaths : undefined,
            });
            setNewMessage('');
            setAttachedImages([]);
            setSelectedFilePaths(new Set());
            setSelectedAgentForNewThread(null);
            setIsAgentProcessing(true); // Agent will start processing
            navigate(`/workspace/${workspaceName}/thread/${newThreadSummary.id}`);
          } catch (err: any) {
            setError(err.message || 'Failed to create thread');
          }
        }
    setIsLoading(false);
  };

  const handleInterruptThread = async () => {
    if (!threadId) return;
    
    setIsInterrupting(true);
    try {
      const result = await api.interruptThread(threadId);
      console.log('Thread interrupted:', result.message);
      setIsAgentProcessing(false); // Agent processing should stop after interrupt
      // Optionally show a success message to the user
    } catch (err: any) {
      console.error('Failed to interrupt thread:', err);
      setError(err.message || 'Failed to interrupt thread');
    }
    setIsInterrupting(false);
  };

  const handleAgentSelectionForNewThread = (agentId: string) => {
    if (!workspaceName) return;
    setSelectedAgentForNewThread(agentId);
    navigate(`/workspace/${workspaceName}`);
  };

  const handleCopyMessage = (msg: ApiChatMessage, msgIndex: number) => {
    let contentToCopy = '';
    if (msg.role === MessageRole.Assistant) {
      msg.content.forEach((block) => {
        contentToCopy += extractTextForCopy(block, messages, workspaceName) + '\n';
        if (block.type === 'tool_use') {
          const nextMessage = messages[msgIndex + 1];
          if (
            nextMessage &&
            nextMessage.role === MessageRole.User
            // Removed: nextMessage.content.every((c) => c.type === 'tool_result')
          ) {
            const resultBlock = nextMessage.content.find(
              (b) => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === (block as ToolUseBlock).id
            ) as ToolResultBlock | undefined;
            if (resultBlock) {
              contentToCopy += extractTextForCopy(resultBlock, messages, workspaceName) + '\n';
            }
          }
        }
      });
    } else {
      contentToCopy = msg.content
        .filter((block) => !(block.meta && block.meta.hidden))
        .map((block) => (block.type === 'text' ? (block as TextBlock).text : '[Non-Text Content]'))
        .join('\n');
    }
    navigator.clipboard
      .writeText(contentToCopy.trim())
      .then(() => {
        setCopiedMessageIndex(msgIndex);
        setTimeout(() => setCopiedMessageIndex(null), 2000);
      })
      .catch((err) => console.error('Failed to copy: ', err));
  };

  if (!currentWorkspace) {
    return (
      <AppLayout>
        <div>Loading workspace data or workspace not selected...</div>
      </AppLayout>
    );
  }
  if (!workspaceName) {
    return (
      <AppLayout>
        <div>Loading workspace name...</div>
      </AppLayout>
    );
  }

  if (!threadId && !selectedAgentForNewThread) {
    return (
      <AppLayout>
        <div className="new-thread-agent-selection">
          <h2>Select an Agent to Start a New Thread</h2>
          {error && <p className="error-message">Error: {error}</p>}
          {availableAgents.length === 0 && !isLoading && (
            <p>
              No agents available.{' '}
              <Link to={`/workspace/${workspaceName}/agents`}>Create one?</Link>
            </p>
          )}
          {isLoading && <p>Loading agents...</p>}
          <div className="agent-list">
            {availableAgents.map((agent) => (
              <button
                key={agent.id}
                onClick={() => handleAgentSelectionForNewThread(agent.id)}
                className="agent-select-button"
              >
                {agent.id} ({agent.llm?.model || 'No model'})
              </button>
            ))}
          </div>
        </div>
      </AppLayout>
    );
  }

  const showChatInterface = threadId || selectedAgentForNewThread;

  const processedMessages = messages.filter((msg) => {
    if (
      msg.role === MessageRole.User &&
      msg.content.length > 0 &&
      msg.content.every((block) => block.type === 'tool_result')
    ) {
      return false;
    }
    if (
      msg.content.length > 0 &&
      msg.content.every((block) => block.meta && block.meta.hidden === true)
    ) {
      return false;
    }
    return true;
  });

  // Find the agent name if we have a thread
  const currentAgentName = currentThread?.agent_id || selectedAgentForNewThread || null;
  const currentAgent = availableAgents.find(a => a.id === currentAgentName);

  return (
    <AppLayout>
      <div className="chat-view-container">
        {/* Display agent info at the top */}
        {currentAgent && (
          <div className="agent-info-header">
            <h3>Agent: {currentAgent.id}</h3>
            <span className="agent-model">Model: {currentAgent.llm?.model || 'Not specified'}</span>
            {currentAgent.llm?.temperature !== undefined && (
              <span className="agent-temperature"> | Temperature: {currentAgent.llm.temperature}</span>
            )}
            {currentAgent.llm?.reasoning?.effort && (
              <span className="agent-reasoning"> | Reasoning: {currentAgent.llm.reasoning.effort}</span>
            )}
          </div>
        )}
        
        {isLoading && !processedMessages.length && !threadId && (
          <div className="loading-chat">Preparing new chat...</div>
        )}
        {isLoading && !!threadId && !processedMessages.length && (
            <div className="loading-chat">Loading messages...</div>
        )}
        {isLoadingFiles && <div className="loading-chat">Loading workspace files...</div>}
        {error && <div className="error-message">Error: {error}</div>}

        <div className="messages-area" ref={chatContainerRef} onScroll={handleScroll}>
          {processedMessages.map((msg, processedMsgIndex) => {
            const idToFind = msg.id;
            let originalMsgRef: ApiChatMessage | undefined = undefined;
            const originalMsgIndex = messages.findIndex((originalMsg, idx) => {
              const idMatch = idToFind !== undefined && idToFind !== null && originalMsg.id === idToFind;
              const refMatch = originalMsg === msg; // Check by reference if IDs are not reliable
              if (idMatch || refMatch) {
                originalMsgRef = originalMsg;
                return true;
              }
              return false;
            });

            console.log(
              `[MainView.map] ProcessedMsg (idx ${processedMsgIndex}, role ${msg.role}, id ${idToFind || 'N/A'}, content: ${msg.content.length} items) -> ` +
              `originalMsgIndex: ${originalMsgIndex}, originalMsg.id: ${originalMsgRef?.id || 'N/A'}`
            );

            if (originalMsgIndex === -1) {
              console.warn('[MainView.map] Could not find original message for processed message:', JSON.parse(JSON.stringify(msg)));
              // Decide how to handle this: skip rendering, or render with a default index?
              // For now, skipping to avoid passing -1 as originalMsgIndex which would break array lookups.
              return null; 
            }

            return (
              <MessageBubble
                key={idToFind || `msg-${processedMsgIndex}-${originalMsgIndex}`} // Improved key for robustness
                msg={msg} // Pass the message from processedMessages
                originalMsgIndex={originalMsgIndex} // Crucial: the index in the *original* messages array
                allMessages={messages} // Pass the full, unfiltered messages array for context
                workspaceName={workspaceName}
                copiedMessageIndex={copiedMessageIndex}
                onCopyMessage={handleCopyMessage}
              />
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        {isScrolledUp && (
          <div className="scroll-buttons">
            <button
              onClick={() =>
                chatContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
              }
            >
              Scroll to Top
            </button>
            <button onClick={scrollToBottom}>Scroll to Bottom</button>
          </div>
        )}

        {showChatInterface && (
          <MessageInputArea 
            newMessage={newMessage}
            setNewMessage={setNewMessage}
            attachedImages={attachedImages}
            setAttachedImages={setAttachedImages}
            selectedFilePaths={selectedFilePaths}
            setSelectedFilePaths={setSelectedFilePaths}
            workspaceFiles={workspaceFiles}
            isLoading={isLoading}
            isLoadingFiles={isLoadingFiles}
            threadId={threadId}
            selectedAgentForNewThread={selectedAgentForNewThread}
            onSendMessage={handleSendMessage}
            onInterruptThread={handleInterruptThread}
            isInterrupting={isInterrupting}
            isAgentProcessing={isAgentProcessing}
            isDragging={isDragging}
            setIsDragging={setIsDragging}
          />
        )}
      </div>
    </AppLayout>
  );
};

export default MainView;