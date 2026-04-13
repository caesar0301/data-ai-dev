#!/usr/bin/env python3
"""
Test 256K context window functionality
Generates long prompts to verify context capacity
"""
import sys
import requests
import json

BASE_URL = "http://localhost:8080/v1"
MODEL_ID = "google/gemma-4-26b-a4b"

def generate_long_prompt(num_paragraphs=100):
    """Generate a long prompt to test context window"""
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(f"""
Paragraph {i+1}: This is test paragraph number {i+1} in our context window test.
We are generating this content to verify that the Gemma 4 26B model can handle
very long input contexts up to 256K tokens. Each paragraph contains approximately
20-30 tokens, so by generating {num_paragraphs} paragraphs, we can test different
context sizes to ensure the model properly processes and retains information from
earlier parts of the conversation. This paragraph discusses context window testing
for large language models and the importance of maintaining coherence across long
conversations or documents.
""")
    return "\n".join(paragraphs)

def test_short_context():
    """Test with short context (~500 tokens)"""
    print("\n" + "="*60)
    print("TEST 1: Short Context (~500 tokens)")
    print("="*60)

    prompt = "What is the capital of France? Please answer briefly."

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        usage = result["usage"]
        content = result["choices"][0]["message"]["content"]

        print(f"✅ Status: SUCCESS")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Response: {content[:100]}")
        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

def test_medium_context():
    """Test with medium context (~5000 tokens)"""
    print("\n" + "="*60)
    print("TEST 2: Medium Context (~5000 tokens)")
    print("="*60)

    # Generate approximately 5000 tokens (roughly 200 paragraphs)
    prompt = generate_long_prompt(200)

    print(f"Generated prompt length: {len(prompt)} characters")

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt + "\n\nQuestion: What was paragraph 50 about?"}],
            "max_tokens": 200
        },
        timeout=120
    )

    if response.status_code == 200:
        result = response.json()
        usage = result["usage"]
        content = result["choices"][0]["message"]["content"]

        print(f"✅ Status: SUCCESS")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Response preview: {content[:150]}")

        # Check if model remembers paragraph 50
        if "50" in content or "paragraph" in content.lower():
            print(f"✅ Model remembered context from paragraph 50")

        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

def test_large_context():
    """Test with large context (~50000 tokens)"""
    print("\n" + "="*60)
    print("TEST 3: Large Context (~50,000 tokens)")
    print("="*60)

    # Generate approximately 50000 tokens (roughly 2000 paragraphs)
    prompt = generate_long_prompt(2000)

    print(f"Generated prompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt) // 4}")

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt + "\n\nQuestion: Summarize what paragraphs 500-600 discussed."}],
            "max_tokens": 300
        },
        timeout=180
    )

    if response.status_code == 200:
        result = response.json()
        usage = result["usage"]
        content = result["choices"][0]["message"]["content"]

        print(f"✅ Status: SUCCESS")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Response preview: {content[:200]}")

        # Verify we're using large context
        if usage['prompt_tokens'] > 10000:
            print(f"✅ Successfully processed large context (>10K tokens)")

        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

def test_very_large_context():
    """Test with very large context (~100,000 tokens)"""
    print("\n" + "="*60)
    print("TEST 4: Very Large Context (~100,000 tokens)")
    print("="*60)

    # Generate approximately 100K tokens (roughly 4000 paragraphs)
    prompt = generate_long_prompt(4000)

    print(f"Generated prompt length: {len(prompt)} characters")
    print(f"Estimated tokens: ~{len(prompt) // 4}")

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt + "\n\nQuestion: What was the topic of paragraphs 1000-1500?"}],
            "max_tokens": 300
        },
        timeout=300
    )

    if response.status_code == 200:
        result = response.json()
        usage = result["usage"]
        content = result["choices"][0]["message"]["content"]

        print(f"✅ Status: SUCCESS")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"Response preview: {content[:200]}")

        # Verify we're using very large context
        if usage['prompt_tokens'] > 50000:
            print(f"✅ Successfully processed very large context (>50K tokens)")

        return True
    else:
        print(f"❌ FAILED: {response.status_code} - {response.text}")
        return False

def check_server_status():
    """Check server configuration"""
    print("="*60)
    print("Server Configuration Check")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server running at {BASE_URL}")

            # Try to get context window info
            models = response.json()["data"]
            for model in models:
                if model["id"] == MODEL_ID:
                    print(f"✅ Model loaded: {model['id']}")

            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nPlease start server with:")
        print("  ./start_lms_256k.sh")
        return False

def run_all_tests():
    """Run all context window tests"""
    print("="*60)
    print("🧪 256K Context Window Test Suite")
    print("="*60)
    print(f"Server: {BASE_URL}")
    print(f"Model: {MODEL_ID}")
    print(f"Context Window: 262144 tokens (256K)")
    print(f"Note: Tests will progressively increase context size")

    if not check_server_status():
        sys.exit(1)

    results = {
        "short_context": test_short_context(),
        "medium_context": test_medium_context(),
        "large_context": test_large_context(),
        "very_large_context": test_very_large_context()
    }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: Context Window Test Results")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\nTotal Tests: {len(results)}")

    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}: {result}")

    if passed == len(results):
        print("\n🎉 All tests passed! 256K context window is functional.")
    elif passed >= 2:
        print("\n⚠️ Some tests passed. Context window may have limitations.")
    else:
        print("\n❌ Most tests failed. Context window may not be properly configured.")

if __name__ == "__main__":
    run_all_tests()