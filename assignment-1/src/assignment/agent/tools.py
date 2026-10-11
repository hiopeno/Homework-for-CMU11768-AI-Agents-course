"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.1.a): Define an OpenAI function-tool schema named ``play_move``.
# It must accept exactly one required string argument named ``move``, explain
# that moves use UCI notation (for example e2e4), and reject extra arguments.
PLAY_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one legal move as White in the current chess game. The server "
            "automatically plays Black's reply when the game does not end."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": (
                        "The move in UCI notation, for example `e2e4`. Include a "
                        "promotion piece suffix when needed, for example `e7e8q`."
                    ),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.3): Define the `simulate_move` tool, like the `play_move` tool.
SIMULATE_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Inspect a chess position or simulate exactly one legal UCI move "
            "without changing the live game. Pass only a complete six-field "
            "FEN to inspect its legal moves, or pass a move such as e2e4 to "
            "inspect the resulting position."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": (
                        "A complete six-field Forsyth-Edwards Notation position."
                    ),
                },
                "move": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional UCI move, such as e2e4 or e7e8q. Use null "
                        "to inspect the supplied position without moving."
                    ),
                },
            },
            # Strict tool schemas require every declared property to be present;
            # null is how the optional move is represented in a call.
            "required": ["fen", "move"],
            "additionalProperties": False,
        },
    },
}

RUN_PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run a Python snippet inside the chess sandbox. The snippet can call "
            "simulate_move(fen, move=None) to explore hypothetical positions and "
            "play_move(move) to make one move on the live board."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source code to execute in the sandbox. "
                        "simulate_move and play_move are already available by name."
                    ),
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
