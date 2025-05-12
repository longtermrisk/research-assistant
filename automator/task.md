# Referencing @files
I want to add a feature that users can refer to files in the current workspace by typing @path/to/file. Here is how it should work:
- the backend has an endpoint GET /workspaces/{workspace_name}/files that returns the files/folder structure of the current workspace. We ignore .gitignored files
- the frontend fetches that list when the workspace is selected
- when the user types @, an in-line menu opens that lets the user select files: ech line has a checkbox, and when the user clicks on a folder, the files in that folder are being shown. (This should follow standard UI practises, i.e. navigation with arrow keys should also work)
- the list of files is being sent as part of the request

Can you implement the frontend part of this? Assume the following folder structure (which would usually be returned by the backend)

automator
├── __init__.py
├── agent.py
├── api
│   ├── __init__.py
│   └── main.py
├── dtypes.py
├── llm.py
├── utils.py
└── workspace.py