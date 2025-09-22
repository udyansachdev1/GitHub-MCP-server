import asyncio
import json
import logging
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    role: str  # 'user' or 'model'
    content: str
    timestamp: datetime
    tools_used: Optional[List[str]] = None

class GeminiMCPChat:
    def __init__(self, gemini_api_key: str = None, repo_key: str = None,
                 mcp_server_command: List[str] = None, mcp_server_env: Dict[str, str] = None,
                 repo_config_path: str = "repo_config.json"):
        """
        Initialize the Gemini + MCP chat system using Gemini's native function calling
        
        Args:
            gemini_api_key: Gemini API key (will use GEMINI_API_KEY env var if not provided)
            repo_key: GitHub token (will use REPO_KEY env var if not provided)
            repo_config_path: Path to JSON file with owner and repo info
            mcp_server_command: Command and args to start MCP server
            mcp_server_env: Additional environment variables for MCP server
        """
        # Get API keys from environment or parameters
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.repo_key = repo_key or os.getenv('REPO_KEY')

        # Load repo config
        self.repo_owner = None
        self.repo_name = None
        try:
            with open(repo_config_path, 'r') as f:
                repo_cfg = json.load(f)
                self.repo_owner = repo_cfg.get('owner')
                self.repo_name = repo_cfg.get('repo')
        except Exception as e:
            logger.warning(f"Could not read repo config: {e}")
        if not self.repo_owner or not self.repo_name:
            logger.warning("Owner or repo not set in repo_config.json. Some tools may require this info.")
        
        if not self.gemini_api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or pass gemini_api_key parameter.")
        
        # Configure Gemini
        genai.configure(api_key=self.gemini_api_key)
        
        # Chat history
        self.conversation_history: List[ChatMessage] = []
        
        # MCP client session
        self.mcp_session: Optional[ClientSession] = None
        self.mcp_server_command = mcp_server_command
        
        # Set up environment variables for MCP server
        self.mcp_server_env = mcp_server_env or {}
        if self.repo_key:
            self.mcp_server_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self.repo_key
        
        self.available_tools: Dict[str, Any] = {}
        self.gemini_tools: List[Tool] = []
        
        # Will be set after MCP initialization
        self.model = None
        self.chat_session = None

    def get_repo_info(self):
        """Return owner and repo name from config"""
        return self.repo_owner, self.repo_name
    
    def _convert_mcp_tool_to_gemini(self, mcp_tool) -> FunctionDeclaration:
        """Convert MCP tool definition to Gemini FunctionDeclaration and log required parameters"""
        # Extract parameters from MCP tool schema
        parameters = {}
        required = []
        if hasattr(mcp_tool, 'inputSchema') and mcp_tool.inputSchema:
            schema = mcp_tool.inputSchema
            if 'properties' in schema:
                parameters = schema['properties']
            if 'required' in schema:
                required = schema['required']
        logger.info(f"Tool '{mcp_tool.name}' required parameters: {required}")
        logger.info(f"Tool '{mcp_tool.name}' properties: {list(parameters.keys())}")
        return FunctionDeclaration(
            name=mcp_tool.name,
            description=mcp_tool.description or f"Execute {mcp_tool.name}",
            parameters={
                "type": "object",
                "properties": parameters,
                "required": required
            }
        )
    
    async def initialize_mcp(self):
        """Initialize MCP connection and set up Gemini with tools"""
        if not self.mcp_server_command:
            logger.info("No MCP server configured, running without tools")
            # Initialize Gemini without tools
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat_session = self.model.start_chat(history=[])
            return
            
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=self.mcp_server_command[0],
                args=self.mcp_server_command[1:],
                env=self.mcp_server_env
            )
            
            # Connect to MCP server
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.mcp_session = session
                    
                    # Initialize the session
                    await session.initialize()
                    
                    # Get available tools
                    tools_result = await session.list_tools()
                    self.available_tools = {
                        tool.name: tool for tool in tools_result.tools
                    }
                    
                    logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
                    
                    # Debug: Print all available tools with their schemas
                    logger.info("Available MCP tools:")
                    for tool_name, tool_info in self.available_tools.items():
                        logger.info(f"  • {tool_name}: {tool_info.description}")
                        if hasattr(tool_info, 'inputSchema') and tool_info.inputSchema:
                            logger.info(f"    Schema: {tool_info.inputSchema}")
                    
                    # Convert MCP tools to Gemini function declarations
                    function_declarations = []
                    for tool_name, tool_info in self.available_tools.items():
                        try:
                            func_decl = self._convert_mcp_tool_to_gemini(tool_info)
                            function_declarations.append(func_decl)
                            logger.info(f"  ✅ Converted {tool_name} to Gemini function")
                        except Exception as e:
                            logger.warning(f"  ⚠️  Failed to convert tool {tool_name}: {e}")
                    
                    # Create Gemini tools
                    if function_declarations:
                        self.gemini_tools = [Tool(function_declarations=function_declarations)]
                        
                        # Initialize Gemini model with tools
                        self.model = genai.GenerativeModel(
                            'gemini-1.5-flash',
                            tools=self.gemini_tools,
                            system_instruction="""You are a helpful GitHub assistant with access to repository tools. 
                            You can list repositories, view files, create issues, and perform other GitHub operations.
                            Use the available tools to help users with their GitHub repositories.
                            Always try to use the appropriate tool when asked about repositories, files, or GitHub operations."""
                        )
                    else:
                        logger.warning("No valid tools found, initializing without tools")
                        self.model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Start chat session
                    self.chat_session = self.model.start_chat(history=[])
                    
                    # Log final setup
                    logger.info(f"Gemini model initialized with {len(function_declarations)} tools")
                    
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            self.mcp_session = None
            # Fallback to model without tools
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.chat_session = self.model.start_chat(history=[])
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute an MCP tool and return the result, auto-filling owner/repo if needed"""
        if not self.mcp_session or tool_name not in self.available_tools:
            return f"Tool '{tool_name}' not available"

        # Auto-fill owner/repo/repository/repo_name if required and missing
        tool_info = self.available_tools[tool_name]
        schema = getattr(tool_info, 'inputSchema', {}) or {}
        props = schema.get('properties', {})
        required = schema.get('required', [])
        owner, repo = self.get_repo_info()
        # Log required and available properties for debugging
        logger.info(f"Executing tool '{tool_name}' with required: {required}, properties: {list(props.keys())}")
        # Common repo parameter names
        repo_param_names = ['repo', 'repository', 'repo_name', 'repository_name']
        if 'owner' in props and not parameters.get('owner'):
            parameters['owner'] = owner
        for param in repo_param_names:
            if param in props and not parameters.get(param):
                parameters[param] = repo

        try:
            result = await self.mcp_session.call_tool(tool_name, parameters)

            # Format the result nicely
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list):
                    formatted_results = []
                    for item in result.content:
                        if hasattr(item, 'text'):
                            formatted_results.append(item.text)
                        else:
                            formatted_results.append(str(item))
                    return "\n".join(formatted_results)
                else:
                    return str(result.content)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return f"Error executing tool {tool_name}: {str(e)}"
    
    async def chat(self, user_message: str) -> str:
        """Process a chat message using Gemini's native function calling"""
        if not self.model or not self.chat_session:
            return "Chat system not initialized. Please run initialize_mcp() first."
        
        # Add user message to history
        self.conversation_history.append(
            ChatMessage(
                role="user",
                content=user_message,
                timestamp=datetime.now()
            )
        )
        
        tools_used = []
        
        try:
            # Send message to Gemini
            response = await self.chat_session.send_message_async(user_message)
            
            # Check if Gemini wants to use function calls
            if response.candidates[0].content.parts:
                final_text_parts = []
                
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Gemini wants to call a function
                        function_call = part.function_call
                        tool_name = function_call.name
                        parameters = dict(function_call.args)
                        
                        logger.info(f"🔧 Executing tool: {tool_name} with params: {parameters}")
                        
                        # Execute the MCP tool
                        tool_result = await self._execute_mcp_tool(tool_name, parameters)
                        tools_used.append(tool_name)
                        
                        # Send the function result back to Gemini
                        function_response = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=tool_name,
                                response={"result": tool_result}
                            )
                        )
                        
                        # Get Gemini's response to the function result
                        follow_up_response = await self.chat_session.send_message_async(
                            genai.protos.Content(parts=[function_response])
                        )
                        
                        # Add the follow-up response text
                        if follow_up_response.text:
                            final_text_parts.append(follow_up_response.text)
                    
                    elif hasattr(part, 'text') and part.text:
                        # Regular text response
                        final_text_parts.append(part.text)
                
                final_response = "\n".join(final_text_parts) if final_text_parts else response.text
            else:
                final_response = response.text
            
            # Add assistant response to history
            self.conversation_history.append(
                ChatMessage(
                    role="model",
                    content=final_response,
                    timestamp=datetime.now(),
                    tools_used=tools_used if tools_used else None
                )
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_response = f"Sorry, I encountered an error: {str(e)}"
            
            # Still add to history for debugging
            self.conversation_history.append(
                ChatMessage(
                    role="model",
                    content=error_response,
                    timestamp=datetime.now()
                )
            )
            
            return error_response
    
    def get_conversation_history(self) -> List[ChatMessage]:
        """Get the full conversation history"""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear conversation history and restart chat session"""
        self.conversation_history.clear()
        if self.model:
            self.chat_session = self.model.start_chat(history=[])
    
    def save_conversation(self, filepath: str):
        """Save conversation to file"""
        conversation_data = []
        for msg in self.conversation_history:
            conversation_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "tools_used": msg.tools_used
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.available_tools.keys())


# Environment variable helpers
def check_environment():
    """Check if required environment variables are set"""
    required_vars = {
        'GEMINI_API_KEY': 'Gemini API key for AI model access',
        'REPO_KEY': 'GitHub Personal Access Token for repository access'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, description))
    
    return missing_vars

def setup_codespaces_environment():
    """Display setup instructions for GitHub Codespaces"""
    print("""
🔧 GitHub Codespaces Setup Instructions

To use this chat in Codespaces, you need to set up these secrets:

1. 🔑 GEMINI_API_KEY:
   - Go to: https://aistudio.google.com/app/apikey
   - Create a new API key
   - Copy the key (starts with 'AI...')

2. 🐙 REPO_KEY:
   - Go to: https://github.com/settings/personal-access-tokens/tokens
   - Generate a new token with these scopes:
     • repo (full repository access)
     • issues (read/write issues)
     • pull_requests (read/write PRs)
     • contents (read repository contents)
   - Copy the token (starts with 'ghp_' or 'github_pat_')

3. 📁 Add to Codespaces Secrets:
   Method A - Repository Secrets (recommended):
   - Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/codespaces
   - Add both secrets there
   
   Method B - User Secrets (works across all codespaces):
   - Go to: https://github.com/settings/codespaces
   - Add secrets in "Repository access" section

4. 🔄 Restart Codespace:
   - After adding secrets, restart your codespace for them to take effect
   - Or run: source ~/.bashrc

Current Environment Status:
""")


# GitHub MCP Setup Helper
async def setup_github_chat():
    """Helper function to set up GitHub MCP chat using environment variables"""
    print("🐙 GitHub MCP Setup with Environment Variables")
    print("=" * 55)
    
    # Check environment variables
    missing_vars = check_environment()
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var, description in missing_vars:
            value = os.getenv(var)
            status = "✅ Set" if value else "❌ Missing"
            print(f"   {var}: {status} - {description}")
        
        print("\n" + "="*55)
        setup_codespaces_environment()
        return None
    

    try:
        # Setup GitHub MCP with environment variables
        chat = GeminiMCPChat(
            mcp_server_command=["npx", "-y", "@modelcontextprotocol/server-github"]
        )
        
        print("\n🔗 Initializing GitHub MCP connection...")
        print("   (This may take a moment to download the MCP server)")
        
        await chat.initialize_mcp()
        
        available_tools = chat.get_available_tools()
        if available_tools:
            print(f"\n✅ Successfully connected! Available GitHub tools ({len(available_tools)}):")
            for tool_name in available_tools[:10]:  # Show first 10 tools
                print(f"   • {tool_name}")
            if len(available_tools) > 10:
                print(f"   ... and {len(available_tools) - 10} more tools")
        else:
            print("\n⚠️  No tools available. Check your environment variables and connection.")
        
        return chat
        
    except Exception as e:
        print(f"\n❌ Failed to initialize chat: {e}")
        return None


# CLI Interface
async def main():
    """Main CLI interface for the chat"""
    print("🚀 Gemini + MCP GitHub Chat Assistant")
    print("   Auto-connecting to GitHub using REPO_KEY")
    print("=" * 50)
    
    # Check if we're likely in Codespaces
    is_codespaces = os.getenv('CODESPACES') == 'true'
    if is_codespaces:
        print(f"🏠 Running in GitHub Codespaces: {os.getenv('CODESPACE_NAME', 'Unknown')}")
    else:
        print("💻 Running in local environment")
    
    # Automatically set up GitHub chat
    print("\n🔗 Setting up GitHub MCP connection...")
    chat = await setup_github_chat()
    
    if not chat:
        print("\n❌ Failed to initialize chat system")
        if is_codespaces:
            print("\n💡 Make sure you've set up your Codespaces secrets correctly:")
            print("   • GEMINI_API_KEY - Get from https://aistudio.google.com/app/apikey")
            print("   • REPO_KEY - Get from https://github.com/settings/personal-access-tokens/tokens")
        return
    
    # Show example prompts for GitHub integration
    if chat.get_available_tools():
        print(f"""
💡 Try asking questions like:
   • "List my repositories"
   • "Show me the README from my main project" 
   • "Create an issue for implementing dark mode"
   • "What are the recent commits in my repo?"
   • "Search for TODO comments in my Python files"
   • "Show me the current repository structure"
   • "What's the latest activity in my repositories?"
        """)
    
    print(f"""
💬 Chat started! Commands:
   • 'quit' or 'exit' - Exit the chat
   • 'clear' - Clear conversation history  
   • 'save' - Save conversation to file
   • 'tools' - List available tools
   • 'env' - Show environment status
""")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            elif user_input.lower() == 'clear':
                chat.clear_history()
                print("🗑️  Conversation history cleared!")
                continue
            elif user_input.lower() == 'env':
                # Show environment status
                missing_vars = check_environment()
                if not missing_vars:
                    print("✅ All required environment variables are set")
                    gemini_key = os.getenv('GEMINI_API_KEY', '')
                    repo_key = os.getenv('REPO_KEY', '')
                    print(f"   GEMINI_API_KEY: {gemini_key[:10]}...")
                    print(f"   REPO_KEY: {repo_key[:10]}...")
                    if is_codespaces:
                        print(f"   CODESPACE_NAME: {os.getenv('CODESPACE_NAME', 'N/A')}")
                else:
                    print("❌ Missing environment variables:")
                    for var, desc in missing_vars:
                        print(f"   {var}: {desc}")
                continue
            elif user_input.lower() == 'tools':
                tools = chat.get_available_tools()
                if tools:
                    print(f"🔧 Available tools ({len(tools)}):")
                    for tool in tools:
                        print(f"   • {tool}")
                else:
                    print("📭 No tools available")
                continue
            elif user_input.lower() == 'save':
                filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                chat.save_conversation(filename)
                print(f"💾 Conversation saved to {filename}")
                continue
            elif not user_input:
                continue
            
            print("\n🤖 Assistant: ", end="", flush=True)
            response = await chat.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    # Check environment and show setup instructions
    missing_vars = check_environment()
    
    if missing_vars:
        print("⚠️  Setup Required!")
        setup_codespaces_environment()
        print("\nCurrent status:")
        for var, description in missing_vars:
            status = "✅" if os.getenv(var) else "❌"
            print(f"  {status} {var}: {description}")
        print("\n" + "="*50)
        print("Please set up the missing environment variables and restart.")
        exit(1)
    
    print("""
🔧 Auto-Setup for GitHub Integration:

✅ Environment Variables:
   - GEMINI_API_KEY: Ready
   - REPO_KEY: Ready

📦 Dependencies (install if needed):
   pip install google-generativeai mcp

🐙 GitHub MCP Integration:
   - Connecting automatically using your REPO_KEY
   - MCP server will be auto-downloaded via npx

🚀 Starting GitHub repository chat...
    """)
    
    # Run the CLI
    asyncio.run(main())