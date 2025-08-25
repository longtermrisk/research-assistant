import React from 'react';
import { Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css'; // Import KaTeX CSS
import {
  ApiChatMessage,
  ContentBlock,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  ImageBlock,
} from '../../types'; // Assuming types are in '../../types'

// Props interface for RenderContentBlock
export interface RenderContentBlockProps {
  block: ContentBlock;
  index: number;
  messages: ApiChatMessage[];
  workspaceName?: string;
}

export const RenderContentBlock: React.FC<RenderContentBlockProps> = ({
  block,
  index,
  messages,
  workspaceName,
}) => {
  if (block.meta && block.meta.hidden === true) {
    return null;
  }

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
    case 'text': {
      const textBlock = block as TextBlock;
      if (textBlock.meta?.display_html && typeof textBlock.meta.display_html === 'string') {
        const htmlContent = textBlock.meta.display_html;

        // Check if it's a full HTML document (likely for Plotly or similar rich HTML outputs)
        if (htmlContent.trim().toLowerCase().startsWith('<!doctype html>')) {
          return (
            <iframe
              key={index}
              className="html-display plotly-iframe" // Specific class for styling if needed
              srcDoc={htmlContent}
              style={{ width: '100%', height: '450px', border: 'none' }} // Initial height, can be adjusted
              sandbox="allow-scripts allow-same-origin" // Essential for Plotly's JavaScript to run
              onLoad={(e) => {
                try {
                  const iframe = e.target as HTMLIFrameElement;
                  setTimeout(() => {
                    try {
                      const body = iframe.contentWindow?.document.body;
                      const htmlDoc = iframe.contentWindow?.document.documentElement;
                      if (body && htmlDoc) {
                        const contentHeight = Math.max(
                          body.scrollHeight,
                          body.offsetHeight,
                          htmlDoc.scrollHeight,
                          htmlDoc.offsetHeight,
                          htmlDoc.clientHeight
                        );
                        if (contentHeight && contentHeight > 0) {
                          console.log('Setting iframe height to:', contentHeight);
                          iframe.style.height = `${Math.max(300, contentHeight)}px`;
                        }
                      }
                    } catch (err) {
                      console.warn('Could not adjust iframe height:', err);
                    }
                  }, 1000); // Added a delay to ensure content is fully rendered
                } catch (err) {
                  console.warn('Error in iframe load handler:', err);
                }
              }}
            />
          );
        } else {
          // For simpler HTML snippets (e.g., Pandas tables), use the existing method
          return (
            <div
              key={index}
              className="text-block html-display"
              dangerouslySetInnerHTML={{ __html: htmlContent }}
            />
          );
        }
      }
      // Enhanced rendering for text content using ReactMarkdown with LaTeX support
      return (
        <div key={index} className="text-block">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
            components={{
              img: () => null // Prevent image rendering
            }}
          >
            {(block as TextBlock).text}
          </ReactMarkdown>
        </div>
      );
    }
    case 'image': {
      const imageBlock = block as ImageBlock;
      return (
        <div key={index} className="image-block">
          <img
            src={`data:${imageBlock.source.media_type};base64,${imageBlock.source.data}`}
            alt="Uploaded content"
          />
        </div>
      );
    }
    case 'tool_use': {
      const toolUseBlock = block as ToolUseBlock;
      return (
        <div key={index} className="tool-use-block">
          <strong>
            Tool Call: {toolUseBlock.name} (ID: {toolUseBlock.id})
          </strong>
          <pre>{JSON.stringify(toolUseBlock.input, null, 2)}</pre>
        </div>
      );
    }
    case 'tool_result': {
      const toolResultBlock = block as ToolResultBlock;
      const correspondingToolUse = findCorrespondingToolUse(toolResultBlock.tool_use_id);
      let resultPrefix = `Result for Tool Call ID: ${toolResultBlock.tool_use_id}`;
      if (correspondingToolUse) {
        resultPrefix = `Result for ${correspondingToolUse.name} (ID: ${toolResultBlock.tool_use_id})`;
      }

      let subagentLinkElement: React.ReactNode = null;
      if (toolResultBlock.meta?.thread_id && workspaceName) {
        const subThreadPath = `/workspace/${workspaceName}/thread/${toolResultBlock.meta.thread_id}`;
        const linkTarget =
          typeof toolResultBlock.meta.message_start === 'number'
            ? `${subThreadPath}#message-${toolResultBlock.meta.message_start}`
            : subThreadPath;
        subagentLinkElement = (
          <div className="subagent-link">
            <Link to={linkTarget}>
              View Subagent Thread: {toolResultBlock.meta.thread_id}
              {typeof toolResultBlock.meta.message_start === 'number' &&
                ` (from message ${toolResultBlock.meta.message_start})`}
            </Link>
          </div>
        );
      }

      return (
        <div key={index} className="tool-result-block">
          <strong>{resultPrefix}</strong>
          {subagentLinkElement}
          {toolResultBlock.content.map((contentItem, idx) => (
            <RenderContentBlock
              key={idx}
              block={contentItem}
              index={idx}
              messages={messages}
              workspaceName={workspaceName}
            />
          ))}
        </div>
      );
    }
    default:      
      return (
        <div key={index} className="unknown-block">
          Unsupported block type
        </div>
      );
  }
};

export default RenderContentBlock;