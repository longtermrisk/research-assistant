import { FileSystemItem } from "../../../types";

export const mockFiles: FileSystemItem[] = [
  {
    id: "automator",
    name: "automator",
    path: "automator",
    type: "folder",
    children: [
      { id: "automator/__init__.py", name: "__init__.py", path: "automator/__init__.py", type: "file" },
      { id: "automator/agent.py", name: "agent.py", path: "automator/agent.py", type: "file" },
      {
        id: "automator/api",
        name: "api",
        path: "automator/api",
        type: "folder",
        children: [
          { id: "automator/api/__init__.py", name: "__init__.py", path: "automator/api/__init__.py", type: "file" },
          { id: "automator/api/main.py", name: "main.py", path: "automator/api/main.py", type: "file" },
        ],
      },
      { id: "automator/dtypes.py", name: "dtypes.py", path: "automator/dtypes.py", type: "file" },
      { id: "automator/llm.py", name: "llm.py", path: "automator/llm.py", type: "file" },
      { id: "automator/utils.py", name: "utils.py", path: "automator/utils.py", type: "file" },
      { id: "automator/workspace.py", name: "workspace.py", path: "automator/workspace.py", type: "file" },
    ],
  },
];