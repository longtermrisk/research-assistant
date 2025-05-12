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
import FileMentionInput from '../../components/FileMentionInput/FileMentionInput'; // Added
import { mockFiles } from '../../components/FileMentionInput/mockData'; // Added

// Interface for an image that has been processed and is ready for display/sending
interface AttachedImage {
  id: string; // For unique key in React lists
  file: File;
  base64: string;
  media_type: string;
}

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
      return toolUseText;
    case 'tool_result':
      const trBlock = block as ToolResultBlock;
      let resultText = `Result for Tool Call ID: ${trBlock.tool_use_id}\n`;
      if (trBlock.meta?.thread_id && currentWorkspaceName) {
        resultText += `(Subagent Thread: /workspace/${currentWorkspaceName}/thread/${trBlock.meta.thread_id}`;
        if (typeof trBlock.meta.message_start === 'number') {
          resultText += `#message-${trBlock.meta.message_start}`;
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
  const [attachedImages, setAttachedImages] = useState<AttachedImage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [availableAgents, setAvailableAgents] = useState<Agent[]>([]);
  const [selectedAgentForNewThread, setSelectedAgentForNewThread] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [mentionedFilePaths, setMentionedFilePaths] = useState<string[]>([]); // Added


  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const chatContainerRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null); // This ref will be passed to FileMentionInput
  const [isScrolledUp, setIsScrolledUp] = useState(false);
  const [copiedMessageIndex, setCopiedMessageIndex] = useState<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);


  // Helper to convert file to base64
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve((reader.result as string).split(',')[1]); // Get only base64 part
      reader.onerror = error => reject(error);
    });
  };

  const handleImageAttach = useCallback(async (files: FileList | null) => {
    if (!files) return;
    const imageFiles = Array.from(files).filter(file => 
      ['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(file.type)
    );

    if (imageFiles.length === 0) return;

    setIsLoading(true); // Indicate processing
    try {
      const newAttachedImages: AttachedImage[] = await Promise.all(
        imageFiles.map(async (file) => {
          const base64 = await fileToBase64(file);
          return {
            id: `${file.name}-${Date.now()}`,
            file,
            base64,
            media_type: file.type,
          };
        })
      );
      setAttachedImages(prev => [...prev, ...newAttachedImages]);
    } catch (err) {
      console.error("Error processing images:", err);
      setError("Failed to attach images.");
    }
    setIsLoading(false);
  }, []);

  const handleRemoveImage = (idToRemove: string) => {
    setAttachedImages(prev => prev.filter(img => img.id !== idToRemove));
  };


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
        setMessages(prevMessages => [...prevMessages, newMessageData]);
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
      setSelectedAgentForNewThread(agentIdFromLocation || null);
    }
  }, [currentWorkspace, workspaceName, threadId, location.state]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages, attachedImages]);

  // Auto-resize for FileMentionInput is handled internally by that component now.
  // Keeping the original textarea auto-resize logic commented out in case of future direct textarea use.
  // useEffect(() => {
  //   if (textareaRef.current) {
  //     textareaRef.current.style.height = 'auto';
  //     textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
  //   }
  // }, [newMessage]);

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

    console.log("Mentioned file paths to be sent (logged):", mentionedFilePaths); // Log mentioned files

    const contentBlocks: ContentBlock[] = [];
    if (newMessage.trim()) {
      // The newMessage already contains the @path/to/file mentions as plain text
      contentBlocks.push({ type: "text", text: newMessage.trim() });
    }
    attachedImages.forEach(img => {
      contentBlocks.push({
        type: "image",
        source: {
          type: "base64",
          media_type: img.media_type,
          data: img.base64,
        } as Base64ImageSource,
      });
    });
    
    // Example of how you might include mentionedFilePaths in the payload if API supported it:
    // const payload: MessagePostPayload = { 
    //   content: contentBlocks,
    //   mentioned_files: mentionedFilePaths // This would require API change
    // };

    if (currentThread && threadId) {
      try {
        await api.postMessage(workspaceName, threadId, { content: contentBlocks });
        setNewMessage('');
        setAttachedImages([]);
        setMentionedFilePaths([]); // Clear mentioned files after sending
      } catch (err: any) {
        setError(err.message || 'Failed to send message');
      }
    } else if (selectedAgentForNewThread) {
      try {
        const newThreadSummary = await api.createThread(workspaceName, {
          agent_id: selectedAgentForNewThread,
          initial_content: contentBlocks, 
        });
        setNewMessage('');
        setAttachedImages([]);
        setMentionedFilePaths([]); // Clear mentioned files after sending
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

  const handleCopyMessage = (msg: ApiChatMessage, msgIndex: number) => {
    let contentToCopy = "";
    if (msg.role === MessageRole.Assistant) {
      msg.content.forEach(block => {
        contentToCopy += extractTextForCopy(block, messages, workspaceName) + "\n";
        if (block.type === 'tool_use') {
          const nextMessage = messages[msgIndex + 1];
          if (nextMessage && nextMessage.role === MessageRole.User && nextMessage.content.every(c => c.type === 'tool_result')) {
            const resultBlock = nextMessage.content.find(b => (b as ToolResultBlock).tool_use_id === block.id) as ToolResultBlock | undefined;
            if (resultBlock) {
              contentToCopy += extractTextForCopy(resultBlock, messages, workspaceName) + "\n";
            }
          } else if (msg.content.some(b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === block.id)){
            // const resultBlock = msg.content.find(b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === block.id) as ToolResultBlock | undefined;
            // if (resultBlock) { /* Already handled by iteration */ }
          }
        }
      });
    } else {
      contentToCopy = msg.content
        .filter(block => block.type === 'text')
        .map(block => (block as TextBlock).text)
        .join('\n');
    }
    navigator.clipboard.writeText(contentToCopy.trim())
      .then(() => {
        setCopiedMessageIndex(msgIndex);
        setTimeout(() => setCopiedMessageIndex(null), 2000);
      })
      .catch(err => console.error("Failed to copy: ", err));
  };

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      handleImageAttach(files);
    }
  }, [handleImageAttach]);

  const handlePaste = useCallback(async (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = event.clipboardData?.items;
    if (items) {
      const files: File[] = [];
      for (let i = 0; i < items.length; i++) {
        if (items[i].kind === 'file' && items[i].type.startsWith('image/')) {
          const file = items[i].getAsFile();
          if (file) {
            files.push(file);
          }
        }
      }
      if (files.length > 0) {
        event.preventDefault(); // Prevent pasting file path as text
        const dataTransfer = new DataTransfer();
        files.forEach(file => dataTransfer.items.add(file));
        await handleImageAttach(dataTransfer.files);
      }
    }
  }, [handleImageAttach]);


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

  const processedMessages = messages.filter(msg => {
    if (msg.role === MessageRole.User && msg.content.length > 0 && msg.content.every(block => block.type === 'tool_result')) {
      return false; 
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
                  if (block.type === 'tool_use') {
                    const toolUse = block as ToolUseBlock;
                    let toolResult: ToolResultBlock | undefined = undefined;
                    const resultInSameMessage = msg.content.find(
                      b => b.type === 'tool_result' && (b as ToolResultBlock).tool_use_id === toolUse.id
                    ) as ToolResultBlock | undefined;
                    if (resultInSameMessage) {
                      toolResult = resultInSameMessage;
                    } else {
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
                  if (block.type === 'tool_result') {
                    let wasHandled = false;
                    for(let i = 0; i < index; i++) {
                        const prevBlock = msg.content[i] as ContentBlock;
                        if(prevBlock.type === 'tool_use' && (prevBlock as ToolUseBlock).id === (block as ToolResultBlock).tool_use_id) {
                            wasHandled = true;
                            break;
                        }
                    }
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
          <div 
            className={`message-input-container ${isDragging ? 'dragging-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {attachedImages.length > 0 && (
              <div className="image-previews-container">
                {attachedImages.map((image, index) => (
                  <div key={image.id} className="image-preview-item">
                    <img src={`data:${image.media_type};base64,${image.base64}`} alt={`preview ${index}`} />
                    <button onClick={() => handleRemoveImage(image.id)} className="remove-image-btn">×</button>
                  </div>
                ))}
              </div>
            )}
            <div className="message-input-area">
              {/* Replaced textarea with FileMentionInput */}
              <FileMentionInput
                value={newMessage}
                onChange={setNewMessage}
                onMentionedFilesChange={setMentionedFilePaths}
                availableFiles={mockFiles} 
                textareaRef={textareaRef}
                onSend={handleSendMessage} 
              />
              {/* The original textarea and paste handler are removed as FileMentionInput handles text input */}
              <input 
                  type="file"
                  multiple 
                  accept="image/png, image/jpeg, image/gif, image/webp"
                  ref={fileInputRef} 
                  style={{ display: 'none' }} 
                  onChange={(e) => handleImageAttach(e.target.files)}
              />
              <button onClick={handleSendMessage} disabled={isLoading || (!newMessage.trim() && attachedImages.length === 0) || (!threadId && !selectedAgentForNewThread)}>
                {isLoading ? 'Sending...' : 'Send'}
              </button>
            </div>
          </div>
        )}
      </div>
    </AppLayout>
  );
};

export default MainView;