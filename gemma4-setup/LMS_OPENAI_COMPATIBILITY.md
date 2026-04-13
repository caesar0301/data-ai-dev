# LM Studio API OpenAI Compatibility Analysis

**Server:** LM Studio CLI (lms) on port 8080  
**Test Model:** google/gemma-4-26b-a4b  
**Analysis Date:** 2026-04-12  

## Executive Summary

LM Studio API provides **partial OpenAI compatibility** with some important differences and limitations. It implements core endpoints and basic functionality but lacks several advanced OpenAI API features.

**Compatibility Score:** ~70% (Core functionality compatible, advanced features missing)

---

## Endpoint Compatibility

### ✅ Fully Compatible Endpoints

| Endpoint | OpenAI Spec | LM Studio Status | Notes |
|----------|-------------|------------------|-------|
| `/v1/models` | ✅ | ✅ | Returns correct structure with `data` array and `object: "list"` |
| `/v1/chat/completions` | ✅ | ✅ | Core functionality works, response structure matches |
| `/v1/completions` (legacy) | ✅ | ✅ | Legacy text completion endpoint works |
| `/v1/embeddings` | ✅ | ✅ | Returns embeddings with correct structure |

### ❌ Missing/Unsupported Endpoints

| Endpoint | OpenAI Spec | LM Studio Status |
|----------|-------------|------------------|
| `/v1/files` | ✅ | ❌ Not implemented |
| `/v1/fine-tuning` | ✅ | ❌ Not implemented |
| `/v1/images/generations` | ✅ | ❌ Not implemented |
| `/v1/audio/*` | ✅ | ❌ Not implemented |
| `/v1/moderations` | ✅ | ❌ Not implemented |
| `/v1/assistants` | ✅ | ❌ Not implemented |
| `/v1/threads` | ✅ | ❌ Not implemented |

---

## Request Parameter Compatibility

### ✅ Supported Parameters

#### Chat Completions Parameters
| Parameter | OpenAI Type | LM Studio Support | Notes |
|-----------|-------------|-------------------|-------|
| `model` | string | ✅ | Required, works correctly |
| `messages` | array | ✅ | Full message array support including system/user/assistant roles |
| `max_tokens` | integer | ✅ | Works correctly, limits generation length |
| `temperature` | number | ✅ | Values 0-2 supported |
| `top_p` | number | ✅ | Nucleus sampling parameter works |
| `stream` | boolean | ✅ | Streaming mode fully functional |
| `stop` | array/string | ✅ | Stop sequences work (tested) |
| `presence_penalty` | number | ✅ | Works (-2.0 to 2.0) |
| `frequency_penalty` | number | ✅ | Works (-2.0 to 2.0) |
| `seed` | integer | ✅ | Parameter accepted (reproducibility unclear) |
| `tools` | array | ✅ | Function calling tools accepted (execution limited) |

### ❌ Unsupported/Ignored Parameters

| Parameter | OpenAI Type | LM Studio Behavior | Impact |
|-----------|-------------|-------------------|---------|
| `logprobs` | boolean | ❌ Returns `null` | No token probability logging |
| `top_logprobs` | integer | ❌ Ignored | Cannot get alternative token probabilities |
| `n` | integer | ❌ Always returns 1 choice | Cannot request multiple completions |
| `response_format` | object | ❌ Ignored/error | Cannot enforce JSON output format |
| `user` | string | ❌ Ignored | No user tracking for abuse monitoring |
| `tool_choice` | string/object | ❌ Ignored | Cannot force tool usage |

---

## Response Structure Analysis

### ✅ Compatible Response Fields

#### Chat Completion Response
```json
{
  "id": "chatcmpl-xxx",              // ✅ Matches OpenAI format
  "object": "chat.completion",       // ✅ Correct object type
  "created": 1775985380,             // ✅ Unix timestamp
  "model": "google/gemma-4-26b-a4b", // ✅ Model identifier
  "choices": [                       // ✅ Array structure correct
    {
      "index": 0,                    // ✅ Choice index
      "message": {                   // ✅ Message structure
        "role": "assistant",         // ✅ Role field
        "content": "...",            // ✅ Content field
        "tool_calls": []             // ✅ Tool calls array
      },
      "finish_reason": "length",     // ✅ Finish reason
      "logprobs": null               // ✅ Field present (but not functional)
    }
  ],
  "usage": {                         // ✅ Token usage tracking
    "prompt_tokens": 24,             // ✅ Input token count
    "completion_tokens": 100,        // ✅ Output token count
    "total_tokens": 124              // ✅ Total tokens
  }
}
```

### ⚠️ LM Studio-Specific Extensions (Non-Standard)

LM Studio adds custom fields not in OpenAI specification:

```json
{
  "system_fingerprint": "google/gemma-4-26b-a4b",  // ⚠️ Custom field (OpenAI uses different format)
  "stats": {},                                     // ⚠️ Custom empty stats object
  "completion_tokens_details": {                   // ⚠️ Custom extension for reasoning models
    "reasoning_tokens": 81                          // ⚠️ Tracks reasoning token usage
  }
}
```

#### Message Extensions for Reasoning Models
```json
{
  "message": {
    "content": "...",
    "reasoning_content": "...",  // ⚠️ LM Studio extension (not in OpenAI spec)
    "tool_calls": []
  }
}
```

**Impact:** The `reasoning_content` field is useful for reasoning models but **breaks OpenAI SDK compatibility** - standard OpenAI clients won't handle this field.

---

## Streaming Format Analysis

### ✅ Streaming Compatibility

LM Studio streaming format matches OpenAI specification:

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","...","delta":{"content":"text"},"finish_reason":null}}
data: [DONE]
```

**Delta Structure:** 
- ✅ Uses `delta` object (correct)
- ✅ `role` appears in first chunk
- ✅ `content` streamed incrementally
- ⚠️ `reasoning_content` streamed (LM Studio extension)

**Streaming Issues:**
- ⚠️ `reasoning_content` in delta is non-standard and may confuse OpenAI SDK clients

---

## Feature Compatibility Matrix

### Core Features

| Feature | OpenAI API | LM Studio | Compatibility |
|---------|------------|-----------|---------------|
| Basic chat completions | ✅ | ✅ | ✅ Full |
| Streaming responses | ✅ | ✅ | ⚠️ Partial (non-standard extensions) |
| System messages | ✅ | ✅ | ✅ Full |
| Conversation history | ✅ | ✅ | ✅ Full |
| Token counting | ✅ | ✅ | ✅ Full |
| Stop sequences | ✅ | ✅ | ✅ Full |
| Temperature control | ✅ | ✅ | ✅ Full |
| Top-p sampling | ✅ | ✅ | ✅ Full |
| Penalty parameters | ✅ | ✅ | ✅ Full |

### Advanced Features

| Feature | OpenAI API | LM Studio | Compatibility |
|---------|------------|-----------|---------------|
| Function calling | ✅ | ⚠️ | ❌ Limited (tools accepted but execution unclear) |
| JSON mode | ✅ | ❌ | ❌ Not supported |
| Multiple completions (n) | ✅ | ❌ | ❌ Always returns 1 choice |
| Log probabilities | ✅ | ❌ | ❌ Returns null |
| Vision/Image inputs | ✅ | ❌ | ❌ Not tested (likely unsupported) |
| Reproducible outputs (seed) | ✅ | ⚠️ | ⚠️ Parameter accepted, effectiveness unclear |
| User tracking | ✅ | ❌ | ❌ Ignored |

---

## SDK Compatibility Testing

### OpenAI Python SDK Compatibility

**Expected compatibility issues:**

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

# ✅ Works - Basic completion
response = client.chat.completions.create(
    model="google/gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Hello"}]
)

# ❌ Won't work - Multiple completions
response = client.chat.completions.create(
    model="google/gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Hello"}],
    n=3  # Will only return 1 choice
)

# ❌ Won't work - JSON mode
response = client.chat.completions.create(
    model="google/gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Return JSON"}],
    response_format={"type": "json_object"}  # Ignored
)

# ❌ Won't work - Logprobs
response = client.chat.completions.create(
    model="google/gemma-4-26b-a4b",
    messages=[{"role": "user", "content": "Test"}],
    logprobs=True,
    top_logprobs=5  # Returns null
)

# ⚠️ Partial - Reasoning content handling
# Standard SDK won't access reasoning_content field
# Custom handling needed:
content = response.choices[0].message.content  # ✅ Works
reasoning = response.choices[0].message.reasoning_content  # ⚠️ SDK may not recognize this field
```

---

## Key Differences Summary

### What Works Well ✅
1. **Basic chat completions** - Core functionality is solid
2. **Message formats** - System/user/assistant roles work correctly
3. **Streaming** - Standard SSE format (with custom extensions)
4. **Token counting** - Accurate usage statistics
5. **Parameter control** - Temperature, top_p, penalties functional
6. **Stop sequences** - Properly stops generation
7. **Model listing** - `/v1/models` endpoint standard
8. **Embeddings** - Full support for embedding generation
9. **Legacy completions** - `/v1/completions` still works

### What's Missing or Different ❌
1. **Multiple completions** - `n` parameter always returns 1 choice
2. **Log probabilities** - No token probability logging
3. **JSON mode** - No `response_format` support
4. **Function calling execution** - Tools accepted but execution limited
5. **Vision/Multimodal** - Image inputs not tested (likely unsupported)
6. **Fine-tuning** - No fine-tuning endpoints
7. **Files management** - No file endpoints
8. **Assistants API** - Not implemented
9. **Moderation** - No content moderation endpoint
10. **Reasoning content** - Non-standard field breaks SDK compatibility

### LM Studio Advantages 🌟
1. **Reasoning model support** - `reasoning_content` field useful for Gemma reasoning models
2. **Reasoning token tracking** - `completion_tokens_details` provides detailed usage
3. **Local inference** - Runs models locally without API costs
4. **Simple setup** - Easy to start with `lms server start`

---

## Recommendations

### For Developers

1. **Use OpenAI SDK cautiously** - Core features work, but handle missing features gracefully
2. **Custom client recommended** - Build wrapper to handle LM Studio-specific fields
3. **Handle reasoning_content** - Extract and use reasoning field for better model understanding
4. **Avoid advanced parameters** - Don't rely on `n`, `logprobs`, `response_format`
5. **Test streaming carefully** - Handle both `content` and `reasoning_content` in delta

### For Production Use

1. **Good for**: Basic chatbots, simple completions, embeddings, reasoning models
2. **Not suitable for**: Applications requiring multiple completions, JSON enforcement, logprobs, vision, assistants
3. **Migration path**: Can use OpenAI SDK for prototyping, but verify feature availability before deployment

---

## OpenAI Python SDK Compatibility Test Results

### Test Summary (Automated SDK Testing)
- **Tests Run**: 9 comprehensive feature tests
- **Passed**: 6 core functionality tests (✅)
- **Partial**: 2 tests with limitations (⚠️)
- **Failed**: 1 advanced feature test (❌)
- **Compatibility Score**: **77.8%** 🟡 Moderate Compatibility

### Detailed Test Results

#### ✅ Fully Passed Tests (6/9)

1. **Basic Chat Completion** ✅
   - Status: SUCCESS
   - Response structure correct
   - Token usage tracking works
   - Reasoning content detected (LM Studio extension)
   
2. **Streaming Completion** ✅
   - Stream format matches OpenAI specification
   - Delta chunks correctly formatted
   - Content streamed incrementally
   - Reasoning chunks present (58 chunks in test)
   
3. **Stop Sequences** ✅
   - Parameters accepted and functional
   - Generation stops correctly when stop sequence matched
   
4. **Penalty Parameters** ✅
   - presence_penalty works (tested 0.5)
   - frequency_penalty works (tested 0.5)
   
5. **Models Endpoint** ✅
   - Returns correct list structure
   - Model IDs and metadata correct
   - Found 3 models available
   
6. **Embeddings** ✅
   - Returns correct embedding structure
   - Vector length: 768 dimensions
   - Object type: "embedding" (correct)

#### ⚠️ Partial Compatibility Tests (2/9)

7. **Multiple Completions (n parameter)** ⚠️
   - Requested: n=3
   - Received: 1 choice only
   - **Limitation**: Parameter ignored, always returns single choice
   
8. **Log Probabilities** ⚠️
   - Parameters accepted (logprobs=True, top_logprobs=5)
   - Returns: None
   - **Limitation**: Feature not implemented

#### ❌ Failed Tests (1/9)

9. **JSON Mode (response_format)** ❌
   - Error: "'response_format.type' must be 'json_schema' or 'text'"
   - **Limitation**: LM Studio has custom response_format interpretation
   - Not compatible with OpenAI's json_object mode

### Additional Manual Tests
```
✅ /v1/completions (legacy) - Legacy endpoint functional
⚠️ tools - Accepted but execution limited (not tested in SDK suite)
⚠️ seed - Parameter accepted, effectiveness unclear (not tested in SDK suite)
```

---

## Version & Configuration

**LM Studio Version**: lms CLI (bundled with LM Studio.app)  
**Server Port**: 8080  
**Default Context**: 4096 tokens  
**Parallel Requests**: 4  
**Loaded Model**: google/gemma-4-26b-a4b (26B-A4B reasoning model)

---

## Conclusion

LM Studio provides **sufficient OpenAI compatibility for basic use cases** but lacks advanced features required for production applications. The non-standard `reasoning_content` extension is valuable for reasoning models but requires custom handling beyond standard OpenAI SDK capabilities.

**Best Use Cases:**
- Local development and testing
- Basic chatbot applications
- Reasoning model experimentation
- Prototyping before OpenAI API deployment

**Limitations:**
- Cannot replace OpenAI API for applications requiring advanced features
- SDK integration requires custom handling for extensions
- Missing features may break existing OpenAI-based applications