import React from 'react';
import {
    ApiChatMessage,
    MessageRole,
    ContentBlock,
    ToolUseBlock,
    ToolResultBlock,
} from '../../types';
import { RenderContentBlock } from './RenderContentBlock';
import './MainView.css';

export interface MessageBubbleProps {
    msg: ApiChatMessage;
    originalMsgIndex: number;
    allMessages: ApiChatMessage[];
    workspaceName?: string;
    copiedMessageIndex: number | null;
    onCopyMessage: (msg: ApiChatMessage, originalMsgIndex: number) => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
    msg,
    originalMsgIndex,
    allMessages,
    workspaceName,
    copiedMessageIndex,
    onCopyMessage,
}) => {
    const visibleContent = msg.content.filter(
        (block) => !(block.meta && block.meta.hidden === true)
    );

    if (visibleContent.length === 0 && msg.role !== MessageRole.System) {
        return null;
    }

    console.log(
        `[MsgBubble] Rendering for originalMsgIndex: ${originalMsgIndex}, role: ${msg.role}`
    );

    return (
        <div
            id={`message-${originalMsgIndex}`}
            className={`message-bubble ${msg.role}`}
        >
            <div className="message-header">
                <span className="role">{msg.role.toUpperCase()}</span>
                <button
                    onClick={() => onCopyMessage(msg, originalMsgIndex)}
                    className="copy-button"
                >
                    {copiedMessageIndex === originalMsgIndex ? 'Copied!' : 'Copy'}
                </button>
            </div>
            <div className="message-content">
                {visibleContent.map((block, itemIndex) => {
                    if (block.type === 'tool_use') {
                        const toolUse = block as ToolUseBlock;
                        console.log(
                            `[MsgBubble ${originalMsgIndex}] Encountered tool_use: ${toolUse.id}`,
                            JSON.parse(JSON.stringify(toolUse))
                        );
                        let toolResult: ToolResultBlock | undefined = undefined;

                        const resultInSameMessage = msg.content.find(
                            (b) =>
                                b.type === 'tool_result' &&
                                (b as ToolResultBlock).tool_use_id === toolUse.id
                        ) as ToolResultBlock | undefined;

                        if (resultInSameMessage) {
                            if (!(resultInSameMessage.meta && resultInSameMessage.meta.hidden)) {
                                toolResult = resultInSameMessage;
                                console.log(
                                    `[MsgBubble ${originalMsgIndex}] Using tool_result from SAME message (not hidden) for TU_ID ${toolUse.id}:`,
                                    JSON.parse(JSON.stringify(toolResult))
                                );
                            } else {
                                console.log(
                                    `[MsgBubble ${originalMsgIndex}] Tool_result in SAME message for TU_ID ${toolUse.id} is HIDDEN.`
                                );
                            }
                        } else {
                            const nextOriginalMessage = allMessages[originalMsgIndex + 1];
                            if (nextOriginalMessage) {
                                if (nextOriginalMessage.role === MessageRole.User) {
                                    console.log(
                                        `[MsgBubble ${originalMsgIndex}] Next message (idx ${originalMsgIndex + 1
                                        }, role User) found. Searching for tool_result for TU_ID ${toolUse.id
                                        }. Content:`,
                                        JSON.parse(JSON.stringify(nextOriginalMessage.content))
                                    );

                                    toolResult = nextOriginalMessage.content.find(
                                        (bInNext, blockIndexInNext) => {
                                            const isTypeToolResult = bInNext.type === 'tool_result';
                                            const trBlock = bInNext as ToolResultBlock;
                                            const isIdMatch = trBlock.tool_use_id === toolUse.id;
                                            const isVisible = !(bInNext.meta && bInNext.meta.hidden);
                                            const overallMatch =
                                                isTypeToolResult && isIdMatch && isVisible;

                                            console.log(
                                                `[MsgBubble ${originalMsgIndex}] TU_ID ${toolUse.id} - Checking block ${blockIndexInNext} in next message:\n` +
                                                `  - Block Type: ${bInNext.type} (Expected: tool_result -> ${isTypeToolResult})\n` +
                                                `  - Block ToolUseID: '${trBlock.tool_use_id}' (Expected: '${toolUse.id}' -> ${isIdMatch})\n` +
                                                `  - Block Visible: ${isVisible} (Expected: true -> ${isVisible})\n` +
                                                `  - Overall match for this block: ${overallMatch}\n` +
                                                `  - Block data:`,
                                                JSON.parse(JSON.stringify(bInNext))
                                            );
                                            return overallMatch;
                                        }
                                    ) as ToolResultBlock | undefined;

                                    if (toolResult) {
                                        console.log(
                                            `[MsgBubble ${originalMsgIndex}] FOUND tool_result in NEXT message for TU_ID ${toolUse.id}:`,
                                            JSON.parse(JSON.stringify(toolResult))
                                        );
                                    } else {
                                        console.log(
                                            `[MsgBubble ${originalMsgIndex}] NOT FOUND - No matching/visible tool_result in NEXT User message for TU_ID ${toolUse.id}.`
                                        );
                                    }
                                } else {
                                    console.log(
                                        `[MsgBubble ${originalMsgIndex}] Next message (idx ${originalMsgIndex + 1
                                        }) is NOT User role (role: ${nextOriginalMessage.role}). Cannot be tool result.`
                                    );
                                }
                            } else {
                                console.log(
                                    `[MsgBubble ${originalMsgIndex}] No next message available to check for tool_result for TU_ID ${toolUse.id}.`
                                );
                            }
                        }

                        if (!toolResult) {
                            console.log(
                                `[MsgBubble ${originalMsgIndex}] No associated visible toolResult found for tool_use ${toolUse.id} to render in cycle.`
                            );
                        }

                        return (
                            <div
                                key={`cycle-${toolUse.id}-${itemIndex}`}
                                className="tool-cycle-block"
                            >
                                <RenderContentBlock
                                    key={`use-${toolUse.id}`}
                                    block={toolUse}
                                    index={0}
                                    messages={allMessages}
                                    workspaceName={workspaceName}
                                />
                                {toolResult && (
                                    <RenderContentBlock
                                        key={`result-${toolUse.id}`}
                                        block={toolResult}
                                        index={1}
                                        messages={allMessages}
                                        workspaceName={workspaceName}
                                    />
                                )}
                            </div>
                        );
                    }

                    if (block.type === 'tool_result') {
                        let wasHandled = false;
                        for (let i = 0; i < itemIndex; i++) {
                            const prevBlock = visibleContent[i] as ContentBlock;
                            if (
                                prevBlock.type === 'tool_use' &&
                                (prevBlock as ToolUseBlock).id ===
                                (block as ToolResultBlock).tool_use_id
                            ) {
                                wasHandled = true;
                                break;
                            }
                        }
                        if (!wasHandled && originalMsgIndex > 0) {
                            const prevOriginalMessage = allMessages[originalMsgIndex - 1];
                            if (prevOriginalMessage.role === MessageRole.Assistant) {
                                const matchingToolUseInPrevMsg = prevOriginalMessage.content.find(
                                    (b) =>
                                        b.type === 'tool_use' &&
                                        (b as ToolUseBlock).id ===
                                        (block as ToolResultBlock).tool_use_id &&
                                        !(b.meta && b.meta.hidden)
                                );
                                if (matchingToolUseInPrevMsg) {
                                    wasHandled = true;
                                }
                            }
                        }
                        if (wasHandled) {
                            console.log(
                                `[MsgBubble ${originalMsgIndex}] Tool_result for ${(block as ToolResultBlock).tool_use_id
                                } (itemIndex ${itemIndex}) was ALREADY HANDLED by a tool_use. Skipping standalone render.`
                            );
                            return null;
                        }
                        console.log(
                            `[MsgBubble ${originalMsgIndex}] Rendering tool_result for ${(block as ToolResultBlock).tool_use_id
                            } (itemIndex ${itemIndex}) as STANDALONE block.`
                        );
                    }

                    return (
                        <RenderContentBlock
                            key={`${block.type}-${(block as any).id || `item-${itemIndex}`}`}
                            block={block}
                            index={itemIndex}
                            messages={allMessages}
                            workspaceName={workspaceName}
                        />
                    );
                })}
            </div>
        </div>
    );
};

export default MessageBubble;