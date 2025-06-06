import React, { useState, useRef, useCallback } from 'react';
import FileMentionInput from './FileMentionInput/FileMentionInput';
import { FileSystemItem } from '../../types';
import './MainView.css';

interface AttachedImage {
  id: string;
  file: File;
  base64: string;
  media_type: string;
}

interface MessageInputAreaProps {
  newMessage: string;
  setNewMessage: (message: string) => void;
  attachedImages: AttachedImage[];
  setAttachedImages: (images: AttachedImage[]) => void;
  selectedFilePaths: Set<string>;
    setSelectedFilePaths: (paths: Set<string>) => void;
  workspaceFiles: FileSystemItem[];
  isLoading: boolean;
  isLoadingFiles: boolean;
  threadId?: string;
  selectedAgentForNewThread: string | null;
  onSendMessage: () => void;
  isDragging: boolean;
  setIsDragging: (isDragging: boolean) => void;
}

const MessageInputArea: React.FC<MessageInputAreaProps> = ({
  newMessage,
  setNewMessage,
  attachedImages,
  setAttachedImages,
  selectedFilePaths,
  setSelectedFilePaths,
  workspaceFiles,
  isLoading,
  isLoadingFiles,
  threadId,
  selectedAgentForNewThread,
  onSendMessage,
  isDragging,
  setIsDragging
}) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve((reader.result as string).split(',')[1]);
      reader.onerror = (error) => reject(error);
    });
  };

  const handleImageAttach = useCallback(
    async (files: FileList | null) => {
      if (!files) return;
      const imageFiles = Array.from(files).filter((file) =>
        ['image/jpeg', 'image/png', 'image/gif', 'image/webp'].includes(file.type)
      );
      if (imageFiles.length === 0) return;

      try {
        const newAttachedImages: AttachedImage[] = await Promise.all(
          imageFiles.map(async (file) => {
            const base64 = await fileToBase64(file);
            return { id: `${file.name}-${Date.now()}`, file, base64, media_type: file.type };
          })
        );
        setAttachedImages((prev) => [...prev, ...newAttachedImages]);
      } catch (err) {
        console.error('Error processing images:', err);
      }
    },
    [setAttachedImages]
  );

  const handleRemoveImage = (idToRemove: string) => {
    setAttachedImages((prev) => prev.filter((img) => img.id !== idToRemove));
  };

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, [setIsDragging]);

  const handleDragLeave = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
  }, [setIsDragging]);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const files = event.dataTransfer.files;
      if (files && files.length > 0) {
        handleImageAttach(files);
      }
    },
    [handleImageAttach, setIsDragging]
  );

  const handlePaste = useCallback(
    async (event: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const items = event.clipboardData?.items;
      if (items) {
        const files: File[] = [];
        for (let i = 0; i < items.length; i++) {
          if (items[i].kind === 'file' && items[i].type.startsWith('image/')) {
            const file = items[i].getAsFile();
            if (file) files.push(file);
          }
        }
        if (files.length > 0) {
          event.preventDefault();
          const dataTransfer = new DataTransfer();
          files.forEach((file) => dataTransfer.items.add(file));
          await handleImageAttach(dataTransfer.files);
        }
      }
    },
    [handleImageAttach]
  );

  return (
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
              <img
                src={`data:${image.media_type};base64,${image.base64}`}
                alt={`preview ${index}`}
              />
              <button
                onClick={() => handleRemoveImage(image.id)}
                className="remove-image-btn"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="message-input-area">
        <FileMentionInput
          value={newMessage}
          onChange={setNewMessage}
          selectedFilePaths={selectedFilePaths}
          onSelectedFilePathsChange={setSelectedFilePaths}
          availableFiles={workspaceFiles}
          textareaRef={textareaRef}
          onSend={onSendMessage}
          onPaste={handlePaste}
        />
        <input
          type="file"
          multiple
          accept="image/png, image/jpeg, image/gif, image/webp"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={(e) => handleImageAttach(e.target.files)}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="attach-button"
          aria-label="Attach images"
          title="Attach images"
        >
          📎
        </button>
        <button
          onClick={onSendMessage}
          disabled={
            isLoading ||
            isLoadingFiles ||
            (!newMessage.trim() && attachedImages.length === 0) ||
            (!threadId && !selectedAgentForNewThread)
          }
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default MessageInputArea;