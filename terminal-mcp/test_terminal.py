import asyncio
import unittest
import sys
import os
from unittest.mock import patch

# Assuming the server code is in a file named terminal_mcp_server.py
# Adjust the import path if your file structure is different.
try:
    # Make sure the server's functions and state are accessible
    from terminal import execute, write_to_stdin, get_logs, sessions
except ImportError:
    print("ERROR: Ensure terminal.py is in the Python path.", file=sys.stderr)
    # As a fallback for execution, try importing directly if in the same directory
    # This might happen in some execution environments.
    try:
        import terminal
        execute = terminal.execute
        write_to_stdin = terminal.write_to_stdin
        get_logs = terminal.get_logs
        sessions = terminal.sessions
    except ImportError as e:
         print(f"Failed to import server components: {e}", file=sys.stderr)
         sys.exit(1) # Exit if server code cannot be imported


class TestTerminalMCPServer(unittest.IsolatedAsyncioTestCase):
    """Test suite for the Terminal MCP Server tools."""

    def setUp(self):
        """Clear the sessions dictionary before each test."""
        sessions.clear()
        print(f"\n--- Starting test: {self._testMethodName} ---")

    async def asyncTearDown(self):
        """Clean up any leftover processes after each test."""
        print(f"--- Tearing down test: {self._testMethodName} ---")
        # Create a copy of keys to avoid modification during iteration
        tab_ids = list(sessions.keys())
        for tab_id in tab_ids:
            session = sessions.get(tab_id)
            if session:
                process = session.get('process')
                if process and process.returncode is None:
                    print(f"Terminating leftover process for tab {tab_id} (PID: {process.pid})", file=sys.stderr)
                    try:
                        process.terminate()
                        # Give it a moment to terminate
                        await asyncio.wait_for(process.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        print(f"Process {process.pid} did not terminate gracefully, killing.", file=sys.stderr)
                        try:
                            process.kill()
                            await asyncio.sleep(0.1) # Short pause after kill
                        except ProcessLookupError:
                             pass # Process already gone
                        except Exception as kill_err:
                            print(f"Error killing process {process.pid}: {kill_err}", file=sys.stderr)
                    except ProcessLookupError:
                         pass # Process already gone
                    except Exception as term_err:
                        print(f"Error terminating process {process.pid}: {term_err}", file=sys.stderr)
                        # Attempt kill as fallback
                        try:
                            process.kill()
                            await asyncio.sleep(0.1)
                        except Exception: pass # Ignore further errors during cleanup

                # Cancel lingering tasks
                stdout_task = session.get('stdout_task')
                stderr_task = session.get('stderr_task')
                if stdout_task and not stdout_task.done():
                    stdout_task.cancel()
                if stderr_task and not stderr_task.done():
                    stderr_task.cancel()

        sessions.clear() # Ensure sessions are cleared
        print(f"--- Finished test: {self._testMethodName} ---")

    async def test_execute_simple_command_finishes(self):
        """Test executing a simple command that finishes before detach."""
        command = "echo Hello MCP World"
        expected_output = "Hello MCP World\n"
        # Use a short detach time, but expect it to finish before that
        result = await execute(command, detach_after_seconds=2.0)

        self.assertIn(expected_output.strip(), result.strip()) # Strip whitespace for robustness
        self.assertNotIn("continues to run in the background", result)
        # Check if session was cleaned up (process set to None)
        self.assertEqual(len(sessions), 1) # Session should still exist for logs
        tab_id = list(sessions.keys())[0]
        self.assertIsNone(sessions[tab_id].get('process'))

    async def test_execute_command_detaches(self):
        """Test executing a command that runs longer than detach time."""
        detach_time = 0.5 # Short detach time
        sleep_time = detach_time + 1.5 # Ensure sleep is longer
        command = f"echo Starting... && sleep {sleep_time} && echo Finished"

        result = await execute(command, detach_after_seconds=detach_time)

        self.assertIn("Starting...", result)
        self.assertNotIn("Finished", result) # Should detach before this prints
        self.assertIn("continues to run in the background in tab:", result)

        # Extract tab_id
        try:
            tab_id = result.split("tab: ")[-1].strip().rstrip(')')
            self.assertTrue(tab_id in sessions)
        except IndexError:
            self.fail("Could not extract tab_id from detach message.")

        # Verify the process is still tracked
        self.assertIsNotNone(sessions[tab_id].get('process'))
        self.assertIsNotNone(sessions[tab_id].get('stdout_task'))

        # Wait for the process to actually finish in the background
        print(f"Waiting for detached process ({tab_id}) to finish...")
        process = sessions[tab_id]['process']
        stdout_task = sessions[tab_id]['stdout_task']
        stderr_task = sessions[tab_id]['stderr_task']
        if process:
            await process.wait() # Wait for the actual process
        if stdout_task:
            await stdout_task # Wait for reader task
        if stderr_task:
            await stderr_task # Wait for reader task


        # Check logs after it should have finished
        await asyncio.sleep(0.1) # Small delay to ensure logs are processed
        final_logs = await get_logs(tab_id, number_of_lines=10)
        self.assertIn("Starting...", final_logs)
        self.assertIn("Finished", final_logs)

    async def test_execute_invalid_command(self):
        """Test executing a command that fails to start."""
        command = "this_command_does_not_exist_12345"
        result = await execute(command, detach_after_seconds=1.0)

        self.assertIn("[MCP_SERVER_ERROR]", result)
        self.assertIn("Failed to start command", result)
        self.assertEqual(len(sessions), 0) # Session should not be created or should be cleaned up

    async def test_get_logs_simple(self):
        """Test retrieving logs."""
        command = "echo Line 1 && echo Line 2 && echo Line 3"
        exec_result = await execute(command, detach_after_seconds=2.0)
        self.assertNotIn("[MCP_SERVER_ERROR]", exec_result)
        self.assertEqual(len(sessions), 1)
        tab_id = list(sessions.keys())[0]

        # Get all logs
        logs_all = await get_logs(tab_id, number_of_lines=10)
        self.assertIn("Line 1", logs_all)
        self.assertIn("Line 2", logs_all)
        self.assertIn("Line 3", logs_all)

        # Get last 2 lines
        logs_last_2 = await get_logs(tab_id, number_of_lines=2)
        self.assertNotIn("Line 1", logs_last_2)
        self.assertIn("Line 2", logs_last_2)
        self.assertIn("Line 3", logs_last_2)

        # Get 0 lines
        logs_0 = await get_logs(tab_id, number_of_lines=0)
        self.assertEqual(logs_0, "")

    async def test_get_logs_non_existent_tab(self):
        """Test getting logs for a tab that doesn't exist."""
        result = await get_logs("invalid-tab-id", number_of_lines=10)
        self.assertIn("[MCP_SERVER_ERROR]", result)
        self.assertIn("Tab ID 'invalid-tab-id' not found", result)

    async def test_write_to_stdin_simple(self):
        """Test writing to stdin of a simple command like cat."""
        # 'cat' reads stdin and echoes it to stdout
        command = "cat"
        detach_time = 0.5
        exec_result = await execute(command, detach_after_seconds=detach_time)
        self.assertIn("continues to run in the background", exec_result)
        tab_id = exec_result.split("tab: ")[-1].strip().rstrip(')')

        # Write to stdin
        input_text = "Hello stdin!\n"
        write_result = await write_to_stdin(tab_id, input_text)
        self.assertIn("Successfully wrote", write_result)

        # Allow time for cat to process and output
        await asyncio.sleep(0.2)

        # Check logs for the echoed input
        logs = await get_logs(tab_id, number_of_lines=10)
        self.assertIn(input_text.strip(), logs.strip()) # cat might add extra newline

        # Close stdin to terminate cat
        session = sessions.get(tab_id)
        process = session.get('process')
        if process and process.stdin and not process.stdin.is_closing():
             try:
                  process.stdin.close()
                  await process.wait() # Wait for cat to exit after stdin closes
             except (BrokenPipeError, ConnectionResetError):
                  pass # Already closed

    async def test_write_to_stdin_non_existent_tab(self):
        """Test writing to stdin for a tab that doesn't exist."""
        result = await write_to_stdin("invalid-tab-id", "some text")
        self.assertIn("[MCP_SERVER_ERROR]", result)
        self.assertIn("Tab ID 'invalid-tab-id' not found", result)

    async def test_write_to_stdin_finished_process(self):
        """Test writing to stdin of a process that has already finished."""
        command = "echo Done"
        exec_result = await execute(command, detach_after_seconds=1.0)
        self.assertNotIn("continues to run", exec_result) # Ensure it finished
        self.assertEqual(len(sessions), 1)
        tab_id = list(sessions.keys())[0]

        # Wait a moment to ensure process is marked finished internally
        await asyncio.sleep(0.1)

        result = await write_to_stdin(tab_id, "too late")
        self.assertIn("[MCP_SERVER_ERROR]", result)
        self.assertIn("is not running", result)

    async def test_get_logs_after_detach_and_finish(self):
        """Test getting logs after a process detaches and then finishes."""
        detach_time = 0.5
        sleep_time = detach_time + 0.5 # Shorter sleep
        command = f"echo DetachStart && sleep {sleep_time} && echo DetachEnd"

        exec_result = await execute(command, detach_after_seconds=detach_time)
        self.assertIn("continues to run in the background", exec_result)
        tab_id = exec_result.split("tab: ")[-1].strip().rstrip(')')

        # Wait longer than the sleep time for the process to finish
        await asyncio.sleep(sleep_time + 0.5)

        # Check logs - should contain start and end messages
        logs = await get_logs(tab_id, number_of_lines=10)
        self.assertIn("DetachStart", logs)
        self.assertIn("DetachEnd", logs)

        # Verify the process is marked as finished in the session state
        self.assertIsNotNone(sessions.get(tab_id))
        self.assertIsNone(sessions[tab_id].get('process')) # Process should be None now


if __name__ == '__main__':
    # Ensure the tests run even if the server script has its own __main__ block
    # We are testing the imported functions, not running the server's main loop.
    unittest.main()
