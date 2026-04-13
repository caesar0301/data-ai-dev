#!/bin/bash
# Start LM Studio server with 256K context window for Gemma 4 26B model
# Usage: ./start_lms_256k.sh

set -e

# Add lms to PATH
export PATH="$HOME/.lmstudio/bin:$PATH"

echo "🚀 Configuring LM Studio server with 256K context window"
echo "📦 Model: google/gemma-4-26b-a4b"
echo "🌐 Server: http://localhost:8080"
echo "📊 Context: 262144 tokens (256K)"
echo ""

# Stop existing server if running
echo "⏹️  Stopping existing server..."
lms server stop 2>/dev/null || echo "   Server already stopped"

# Unload existing model if loaded
echo "⏹️  Unloading existing model..."
lms unload google/gemma-4-26b-a4b 2>/dev/null || echo "   Model already unloaded"

# Load model with 256K context window
echo ""
echo "⏳ Loading model with 256K context window..."
lms load google/gemma-4-26b-a4b \
  --context-length 262144 \
  --parallel 4 \
  --gpu max \
  -y

# Start server on port 8080
echo ""
echo "🌐 Starting server on port 8080..."
lms server start --port 8080

# Verify configuration
echo ""
echo "✅ Server configuration:"
echo ""
lms ps

echo ""
echo "✅ Server is ready!"
echo ""
echo "Endpoints:"
echo "  - http://localhost:8080/v1/chat/completions"
echo "  - http://localhost:8080/v1/models"
echo "  - http://localhost:8080/v1/embeddings"
echo ""
echo "Model identifier: google/gemma-4-26b-a4b"
echo "Context window: 262144 tokens (256K)"
echo "Parallel requests: 4"
echo "GPU: max offload"
echo ""
echo "Press Ctrl+C to stop the server"