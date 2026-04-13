#!/usr/bin/env python3
"""
Test LM Studio API compatibility using official OpenAI Python SDK
Validates endpoint compatibility and identifies limitations
"""
import sys

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI SDK not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

BASE_URL = "http://localhost:8080/v1"
MODEL_ID = "google/gemma-4-26b-a4b"

def test_basic_completion():
    """Test basic chat completion"""
    print("\n" + "="*60)
    print("TEST 1: Basic Chat Completion")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=100
        )

        print(f"✅ Status: SUCCESS")
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        print(f"Finish Reason: {response.choices[0].finish_reason}")
        print(f"Content: {response.choices[0].message.content[:100]}")

        # Check for reasoning_content (LM Studio extension)
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"⚠️ Found 'reasoning_content' (LM Studio extension)")
            reasoning = response.choices[0].message.reasoning_content
            if reasoning:
                print(f"Reasoning Preview: {reasoning[:80]}...")

        # Token usage
        print(f"\n📊 Token Usage:")
        print(f"  Prompt: {response.usage.prompt_tokens}")
        print(f"  Completion: {response.usage.completion_tokens}")
        print(f"  Total: {response.usage.total_tokens}")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_streaming():
    """Test streaming completion"""
    print("\n" + "="*60)
    print("TEST 2: Streaming Completion")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Count from 1 to 5"}],
            max_tokens=100,
            stream=True
        )

        print("✅ Stream started")
        print("🤖 Response: ", end="", flush=True)

        content_parts = []
        reasoning_parts = []

        for chunk in stream:
            delta = chunk.choices[0].delta

            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_parts.append(delta.reasoning_content)
                # Don't print reasoning in streaming to keep output clean

            if delta.content:
                content_parts.append(delta.content)
                print(delta.content, end="", flush=True)

        print()  # Newline

        if reasoning_parts:
            print(f"⚠️ Reasoning detected (LM Studio extension): {len(reasoning_parts)} chunks")

        print(f"✅ Status: SUCCESS")
        print(f"Content length: {len(''.join(content_parts))} chars")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_multiple_completions():
    """Test n parameter for multiple completions"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Completions (n parameter)")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=20,
            n=3  # Request 3 different completions
        )

        print(f"Requested n=3 completions")
        print(f"Received: {len(response.choices)} choices")

        if len(response.choices) == 1:
            print(f"⚠️ PARTIAL: n parameter ignored, only returned 1 choice")
            print("This is expected for LM Studio (OpenAI limitation)")
            return "partial"
        else:
            print(f"✅ SUCCESS: Got {len(response.choices)} choices")
            for i, choice in enumerate(response.choices):
                print(f"  Choice {i}: {choice.message.content}")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_logprobs():
    """Test logprobs parameter"""
    print("\n" + "="*60)
    print("TEST 4: Log Probabilities")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=20,
            logprobs=True,
            top_logprobs=5
        )

        logprobs_result = response.choices[0].logprobs

        if logprobs_result is None:
            print(f"⚠️ PARTIAL: logprobs parameter accepted but returns None")
            print("This is expected for LM Studio (feature not implemented)")
            return "partial"
        else:
            print(f"✅ SUCCESS: logprobs returned")
            print(f"  Content: {logprobs_result.content}")
            return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_json_mode():
    """Test response_format for JSON mode"""
    print("\n" + "="*60)
    print("TEST 5: JSON Mode (response_format)")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Return a JSON object with name and age"}],
            max_tokens=50,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        # Try to parse as JSON
        import json
        try:
            parsed = json.loads(content)
            print(f"✅ SUCCESS: Response is valid JSON")
            print(f"  JSON: {parsed}")
            return True
        except json.JSONDecodeError:
            print(f"⚠️ PARTIAL: response_format accepted but output not valid JSON")
            print(f"  Content: {content}")
            print("This is expected for LM Studio (JSON mode not enforced)")
            return "partial"
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_stop_sequences():
    """Test stop sequences"""
    print("\n" + "="*60)
    print("TEST 6: Stop Sequences")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Say hello world"}],
            max_tokens=50,
            stop=["hello", "Hello"]
        )

        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        print(f"✅ Status: SUCCESS")
        print(f"Finish Reason: {finish_reason}")
        print(f"Content: {content}")

        if finish_reason == "stop":
            print("✅ Stop sequence triggered correctly")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_penalty_parameters():
    """Test presence_penalty and frequency_penalty"""
    print("\n" + "="*60)
    print("TEST 7: Penalty Parameters")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Test penalty"}],
            max_tokens=30,
            presence_penalty=0.5,
            frequency_penalty=0.5
        )

        print(f"✅ Status: SUCCESS")
        print(f"Parameters accepted:")
        print(f"  presence_penalty=0.5")
        print(f"  frequency_penalty=0.5")
        print(f"Content: {response.choices[0].message.content}")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_models_endpoint():
    """Test models listing"""
    print("\n" + "="*60)
    print("TEST 8: Models Endpoint")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        models = client.models.list()

        print(f"✅ Status: SUCCESS")
        print(f"Models available:")

        for model in models.data:
            print(f"  - {model.id} (owned by: {model.owned_by})")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_embeddings():
    """Test embeddings endpoint"""
    print("\n" + "="*60)
    print("TEST 9: Embeddings")
    print("="*60)

    client = OpenAI(base_url=BASE_URL, api_key="dummy")

    try:
        response = client.embeddings.create(
            model="text-embedding-nomic-embed-text-v1.5",
            input="test text for embedding"
        )

        print(f"✅ Status: SUCCESS")
        print(f"Embedding vector length: {len(response.data[0].embedding)}")
        print(f"Object type: {response.data[0].object}")
        print(f"First 5 values: {response.data[0].embedding[:5]}")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def run_all_tests():
    """Run all compatibility tests"""
    print("="*60)
    print("🧪 LM Studio OpenAI SDK Compatibility Test Suite")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Model: {MODEL_ID}")
    print(f"SDK: OpenAI Python SDK")

    results = {
        "basic_completion": test_basic_completion(),
        "streaming": test_streaming(),
        "multiple_completions": test_multiple_completions(),
        "logprobs": test_logprobs(),
        "json_mode": test_json_mode(),
        "stop_sequences": test_stop_sequences(),
        "penalty_parameters": test_penalty_parameters(),
        "models_endpoint": test_models_endpoint(),
        "embeddings": test_embeddings()
    }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Compatibility Test Results")
    print("="*60)

    passed = sum(1 for v in results.values() if v == True)
    partial = sum(1 for v in results.values() if v == "partial")
    failed = sum(1 for v in results.values() if v == False)

    print(f"\n✅ Passed: {passed}")
    print(f"⚠️ Partial: {partial}")
    print(f"❌ Failed: {failed}")
    print(f"\nTotal Tests: {len(results)}")

    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅" if result == True else ("⚠️" if result == "partial" else "❌")
        print(f"  {status} {test_name}: {result}")

    # Overall compatibility score
    compatibility = (passed + (partial * 0.5)) / len(results) * 100
    print(f"\n📊 Compatibility Score: {compatibility:.1f}%")

    if compatibility >= 80:
        print("🟢 HIGH COMPATIBILITY - Suitable for most applications")
    elif compatibility >= 60:
        print("🟡 MODERATE COMPATIBILITY - Good for basic use, limitations exist")
    elif compatibility >= 40:
        print("🟠 LOW COMPATIBILITY - Limited features, use with caution")
    else:
        print("🔴 MINIMAL COMPATIBILITY - Not recommended for production")

    return results


if __name__ == "__main__":
    # Check server availability
    import requests
    try:
        requests.get(f"{BASE_URL.replace('/v1', '')}/v1/models", timeout=5)
        print(f"✅ LM Studio server running at {BASE_URL}")
    except:
        print(f"❌ Cannot connect to {BASE_URL}")
        print("Please start LM Studio server first:")
        print("  export PATH=\"$HOME/.lmstudio/bin:$PATH\"")
        print("  lms server start --port 8080")
        print("  lms load google/gemma-4-26b-a4b")
        sys.exit(1)

    run_all_tests()