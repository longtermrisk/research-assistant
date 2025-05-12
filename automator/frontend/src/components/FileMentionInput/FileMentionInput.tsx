import React, { useState, useEffect, useRef, useCallback } from 'react';
import { FileSystemItem } from '../../../types';
import styles from './FileMentionInput.module.css';

interface FileMentionInputProps {
  value: string;
  onChange: (value: string) => void;
  onMentionedFilesChange: (paths: string[]) => void;
  availableFiles: FileSystemItem[];
  textareaRef?: React.RefObject<HTMLTextAreaElement>;
  onSend?: () => void;
}

interface DisplayItem extends FileSystemItem {
  level: number;
}

const getAllFilePathsInFolder = (folderItem: FileSystemItem): string[] => {
  let paths: string[] = [];
  if (folderItem.type === 'folder' && folderItem.children) {
    for (const child of folderItem.children) {
      if (child.type === 'file') {
        paths.push(child.path);
      } else if (child.type === 'folder') {
        paths = paths.concat(getAllFilePathsInFolder(child));
      }
    }
  }
  return paths;
};

const FileMentionInput: React.FC<FileMentionInputProps> = ({
  value,
  onChange,
  onMentionedFilesChange,
  availableFiles,
  textareaRef: externalTextareaRef,
  onSend,
}) => {
  const [showPopup, setShowPopup] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredDisplayItems, setFilteredDisplayItems] = useState<DisplayItem[]>([]);
  const [activeIndex, setActiveIndex] = useState(0);
  const [openFolders, setOpenFolders] = useState<Set<string>>(new Set());
  const [selectedFilePaths, setSelectedFilePaths] = useState<Set<string>>(new Set());
  const [mentionStartIndex, setMentionStartIndex] = useState<number | null>(null);

  const internalTextareaRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = externalTextareaRef || internalTextareaRef;
  const popupRef = useRef<HTMLDivElement>(null);
  const activeItemRef = useRef<HTMLLIElement | null>(null);

  const findItemByIdRecursive = (id: string, items: FileSystemItem[]): FileSystemItem | null => {
    for (const item of items) {
      if (item.id === id) return item;
      if (item.children) {
        const foundInChild = findItemByIdRecursive(id, item.children);
        if (foundInChild) return foundInChild;
      }
    }
    return null;
  };

  const getFolderSelectionState = useCallback((folderDisplayItem: DisplayItem): 'all' | 'some' | 'none' => {
    const originalFolderItem = findItemByIdRecursive(folderDisplayItem.id, availableFiles);
    if (!originalFolderItem || originalFolderItem.type !== 'folder') return 'none';

    const allDescendantFiles = getAllFilePathsInFolder(originalFolderItem);
    if (allDescendantFiles.length === 0) return 'none';

    let selectedCount = 0;
    for (const filePath of allDescendantFiles) {
      if (selectedFilePaths.has(filePath)) {
        selectedCount++;
      }
    }

    if (selectedCount === 0) return 'none';
    if (selectedCount === allDescendantFiles.length) return 'all';
    return 'some';
  }, [selectedFilePaths, availableFiles]);

  const flattenAndFilter = useCallback(
    (items: FileSystemItem[], currentSearchTerm: string, currentOpenFolders: Set<string>, level = 0): DisplayItem[] => {
      let result: DisplayItem[] = [];
      const lowerSearchTerm = currentSearchTerm.toLowerCase();
      for (const item of items) {
        const matchesSearch = item.name.toLowerCase().includes(lowerSearchTerm) || item.path.toLowerCase().includes(lowerSearchTerm);
        if (item.type === 'folder') {
          if (matchesSearch || currentOpenFolders.has(item.id) || !currentSearchTerm.trim()) {
            result.push({ ...item, level });
            if (currentOpenFolders.has(item.id) && item.children) {
              const childSearchTerm = matchesSearch ? currentSearchTerm : "";
              result = result.concat(flattenAndFilter(item.children, childSearchTerm, currentOpenFolders, level + 1));
            }
          }
        } else if (matchesSearch || !currentSearchTerm.trim()) {
          result.push({ ...item, level });
        }
      }
      return result;
    },
    []
  );

  useEffect(() => {
    if (showPopup) {
      const items = flattenAndFilter(availableFiles, searchTerm, openFolders);
      setFilteredDisplayItems(items);
      setActiveIndex(items.length > 0 ? 0 : -1);
    }
  }, [searchTerm, availableFiles, openFolders, showPopup, flattenAndFilter]);

  useEffect(() => {
    onMentionedFilesChange(Array.from(selectedFilePaths));
  }, [selectedFilePaths, onMentionedFilesChange]);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    const cursorPos = e.target.selectionStart;
    onChange(text);

    let atIndex = -1;
    for (let i = cursorPos - 1; i >= 0; i--) {
      if (text[i] === '@') {
        if (i === 0 || /\s/.test(text[i - 1]) || ['(', '['].includes(text[i-1])) {
          atIndex = i;
          break;
        }
      }
      if (/\s/.test(text[i]) && i < cursorPos -1) break;
    }

    if (atIndex !== -1) {
      const currentMention = text.substring(atIndex + 1, cursorPos);
      if (/^[a-zA-Z0-9/._-]*$/.test(currentMention)) {
        setSearchTerm(currentMention);
        setShowPopup(true);
        setMentionStartIndex(atIndex);
      } else {
        closePopup(); // Close if invalid characters after @
      }
    } else {
      closePopup();
    }
  };

  const closePopup = useCallback(() => {
    setShowPopup(false);
    // Keep activeIndex and searchTerm for potential quick reopen, reset mentionStartIndex
    setMentionStartIndex(null); 
  }, []);

  const toggleFolderExpansion = (folderId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setOpenFolders(prev => {
      const newSet = new Set(prev);
      if (newSet.has(folderId)) newSet.delete(folderId);
      else newSet.add(folderId);
      return newSet;
    });
  };

  const insertMentionText = (textToInsert: string) => {
    if (mentionStartIndex !== null) {
      const textBefore = value.substring(0, mentionStartIndex);
      const currentCursorPos = textareaRef.current?.selectionStart || value.length;
      const textAfter = value.substring(currentCursorPos);
      
      const newText = `${textBefore}@${textToInsert} ${textAfter}`;
      onChange(newText);
      
      setTimeout(() => {
        const newCursorPos = (textBefore + "@" + textToInsert + " ").length;
        textareaRef.current?.focus();
        textareaRef.current?.setSelectionRange(newCursorPos, newCursorPos);
      }, 0);
    }
    closePopup();
  };
  
  const handleCheckboxChange = (item: DisplayItem, e: React.ChangeEvent<HTMLInputElement>) => {
    e.stopPropagation();
    const newSelectedPaths = new Set(selectedFilePaths);
    if (item.type === 'file') {
      if (e.target.checked) newSelectedPaths.add(item.path);
      else newSelectedPaths.delete(item.path);
    } else if (item.type === 'folder') {
      const originalFolderItem = findItemByIdRecursive(item.id, availableFiles);
      if (originalFolderItem) {
        const filesInFolder = getAllFilePathsInFolder(originalFolderItem);
        if (e.target.checked) filesInFolder.forEach(p => newSelectedPaths.add(p));
        else filesInFolder.forEach(p => newSelectedPaths.delete(p));
      }
    }
    setSelectedFilePaths(newSelectedPaths);
  };

  const handleItemAction = (item: DisplayItem) => {
    insertMentionText(item.path);

    const newSelectedPaths = new Set(selectedFilePaths);
    if (item.type === 'file') {
      newSelectedPaths.add(item.path); // Ensure file is selected
    } else if (item.type === 'folder') {
      const originalFolderItem = findItemByIdRecursive(item.id, availableFiles);
      if (originalFolderItem) { // Select all files in folder
        const filesInFolder = getAllFilePathsInFolder(originalFolderItem);
        filesInFolder.forEach(p => newSelectedPaths.add(p));
      }
    }
    setSelectedFilePaths(newSelectedPaths);
  };

  const handleKeyDownTextarea = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showPopup && filteredDisplayItems.length > 0 && activeIndex !== -1) {
      if (['ArrowUp', 'ArrowDown', 'Enter', 'Escape', 'Tab'].includes(e.key)) {
        e.preventDefault();
        handleKeyDownPopup(e);
      }
    } else if (e.key === 'Enter' && !e.shiftKey && onSend) {
      e.preventDefault();
      onSend();
    } else if (e.key === ' ' && showPopup && mentionStartIndex !== null) {
        // If space is typed while popup is open, commit the current search term as a literal mention
        const currentMentionText = value.substring(mentionStartIndex + 1, textareaRef.current?.selectionStart);
        if (currentMentionText.trim()) {
             // No specific item selected from list, so just close popup.
             // The text already typed (@searchterm) remains.
        }
        closePopup(); // Close popup on space
    }
  };

  const handleKeyDownPopup = (e: React.KeyboardEvent) => {
    if (!showPopup || filteredDisplayItems.length === 0) return;
    let newActiveIndex = activeIndex;

    switch (e.key) {
      case 'ArrowDown':
        newActiveIndex = (activeIndex + 1) % filteredDisplayItems.length;
        setActiveIndex(newActiveIndex);
        break;
      case 'ArrowUp':
        newActiveIndex = (activeIndex - 1 + filteredDisplayItems.length) % filteredDisplayItems.length;
        setActiveIndex(newActiveIndex);
        break;
      case 'Enter':
      case 'Tab':
        if (activeIndex >= 0 && filteredDisplayItems[activeIndex]) {
          handleItemAction(filteredDisplayItems[activeIndex]);
        }
        break;
      case 'Escape':
        closePopup();
        break;
    }
  };

  useEffect(() => {
    if (activeItemRef.current) {
      activeItemRef.current.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
  }, [activeIndex]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(event.target as Node) &&
          textareaRef.current && !textareaRef.current.contains(event.target as Node)) {
        closePopup();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [closePopup, textareaRef]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.max(scrollHeight, 40)}px`;
    }
  }, [value, textareaRef]);

  return (
    <div className={styles.container}>
      {showPopup && (
        <div
          ref={popupRef}
          className={styles.popup}
          style={{
            bottom: textareaRef.current ? textareaRef.current.offsetHeight + 5 : '100%',
            left: 0,
            width: textareaRef.current ? textareaRef.current.offsetWidth : 'auto',
          }}
        >
          {filteredDisplayItems.length > 0 ? (
            <ul>
              {filteredDisplayItems.map((item, index) => {
                const folderState = item.type === 'folder' ? getFolderSelectionState(item) : 'none';
                const isChecked = item.type === 'file' ? selectedFilePaths.has(item.path) : folderState === 'all';
                const isIndeterminate = item.type === 'folder' && folderState === 'some';
                return (
                  <li
                    key={item.id}
                    ref={index === activeIndex ? activeItemRef : null}
                    className={`${styles.popupItem} ${index === activeIndex ? styles.active : ''}`}
                    onClick={() => handleItemAction(item)}
                    style={{ paddingLeft: `${10 + item.level * 15}px` }}
                  >
                    <span
                      className={styles.folderToggle}
                      onClick={(e) => toggleFolderExpansion(item.id, e)}
                    >
                      {item.type === 'folder' ? (openFolders.has(item.id) ? '▼' : '►') : <span style={{ display: 'inline-block', width: '10px' }}>&nbsp;</span>}
                    </span>
                    <input
                      type="checkbox"
                      className={styles.checkbox}
                      checked={isChecked}
                      ref={el => el && (el.indeterminate = isIndeterminate)}
                      onChange={(e) => handleCheckboxChange(item, e)}
                    />
                    <span className={item.type === 'folder' ? styles.folderName : styles.fileName}>
                      {item.name} <span className={styles.itemPath}>({item.path})</span>
                    </span>
                  </li>
                );
              })}
            </ul>
          ) : (
            searchTerm && <div className={styles.noResults}>No files found for "{searchTerm}"</div>
          )}
        </div>
      )}
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleTextChange}
        onKeyDown={handleKeyDownTextarea}
        className={styles.textarea}
        placeholder="Type your message, or @ to mention files..."
        rows={1}
      />
    </div>
  );
};

export default FileMentionInput;