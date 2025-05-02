import asyncio
from automator.workspace import Workspace


async def main():
    workspace = Workspace('my-workspace')

    agents = workspace.list_agents(limit=20)
    print("Agents:\n", agents, "\n", "-" * 20)
    threads = workspace.list_threads(limit=20)
    print("Threads:\n", threads, "\n", "-" * 20)

    bash_agent = workspace.get_agent('bash')
    thread = workspace.get_thread('example-thread')

    with open('example.md', 'w') as f:
        f.write(thread.to_markdown())

    # Continue the thread
    thread = await thread.run(input("Query> "))
    async for message in thread:
        print(message)
    with open('example.md', 'w') as f:
        f.write(thread.to_markdown())

    await thread.cleanup()

if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())