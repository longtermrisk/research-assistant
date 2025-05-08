# GUI for automator
Automator is currently a python library for creating and running agents, and handling workspaces. Now, we want a to implement a GUI for this system as a webapp.

The GUI should consist of a fastapi backend and a react-based frontend. The style should be a modern dark-themed design that appeals to coders, inspired by VSCode.

Views:
# Initial view: select or create a workspace

# Main view
- Top bar
  - shows workspace switcher
  - button "Agents" leads to Agent screen
- Left sidebar shows threads of the project
  - sidebar can be hidden or expanded
  - show the first words of the first user message in a thread on the thread button
  - Top button shows "New thread" button
- New thread
  - clicking on the button expands the list of agents that exist in the workspace
  - when an agent is selected, an empty thread will be displayed in the main area, allowing the user to type their first message
  - once the message is sent, the actual backend request will be sent that creates a new thread with the selected agent and initial message
- Main area shows the chat
  - user message field is always on the bottom
    - single line, grows when multiline input is entered
    - copy&paste images or drag&drop images add images to the user message, which show up on top of the text area as mini-preview
  - messages are being displayed in the main area
    - messages have a top bar showing message role + expand/minimize button + copy to clipboard button
    - expanded messages show the list of content blocks
    - we need components for each content block type implemented in dtypes.py
      - text blocks are by default rendered as markdown
      - tool results block are displayed below their corresponding tool_use block
      - subagent calls render a link that displays the subagent thread in the gui
  - new messages are sent from the backend to the frontend via SSE
  - scrolling:
    - when the chat area is scrolled down and a new message arrives, it automatically scrolls down to the end again
    - if the user has scrolled up, arrows appear to scroll all the way up or down
  
# Agent screen
- left sidebar and top bar are the same as in the main view
- Main area: shows list of existing agents, "Create new" button"

# Notes
- Avoid overly verbose code - for example, don't recreate all pydantic models that are already implemented in dtypes.py if you can reuse them
- Test early and simple - e.g. you can test the backend via curl (doesn't need to be pytest, unless it's easier)