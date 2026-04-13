#!/usr/bin/env python3
"""
Comprehensive generation tests for MLX-VLM models
Compares performance across different scenarios
"""
import requests
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:8080/v1"

class GenerationTester:
    def __init__(self, model_id):
        self.model_id = model_id
        self.results = []

    def test_generation(self, test_name, prompt, max_tokens=50, temperature=1.0, top_p=0.95, stream=False):
        """Run a generation test and collect metrics"""
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        print(f"Model: {self.model_id}")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Max tokens: {max_tokens}, Temperature: {temperature}, Top-p: {top_p}")
        print(f"Streaming: {stream}")

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": stream
        }

        start_time = time.time()

        try:
            if stream:
                response_text = self._test_streaming(payload)
            else:
                response_text = self._test_non_streaming(payload)

            elapsed_time = time.time() - start_time

            print(f"\n✅ Test completed in {elapsed_time:.2f}s")
            print(f"\n🤖 Generated text:\n{response_text}")

            # Collect result
            result = {
                "test_name": test_name,
                "prompt_length": len(prompt),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "streaming": stream,
                "elapsed_time": elapsed_time,
                "response_length": len(response_text)
            }
            self.results.append(result)
            return result

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return None

    def _test_non_streaming(self, payload):
        """Test non-streaming generation"""
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        result = response.json()

        # Print metrics
        if "usage" in result:
            usage = result["usage"]
            print(f"\n📊 Metrics:")
            print(f"  Input tokens: {usage.get('prompt_tokens', usage.get('input_tokens', 'N/A'))}")
            print(f"  Output tokens: {usage.get('completion_tokens', usage.get('output_tokens', 'N/A'))}")
            if "completion_tokens_details" in usage:
                details = usage["completion_tokens_details"]
                print(f"  Reasoning tokens: {details.get('reasoning_tokens', 'N/A')}")
            print(f"  Prefill t/s: {usage.get('prompt_tps', 'N/A')}")
            print(f"  Decode t/s: {usage.get('generation_tps', 'N/A')}")
            print(f"  Peak memory (GB): {usage.get('peak_memory', 'N/A')}")

        # Handle reasoning models that use reasoning_content field
        message = result["choices"][0]["message"]
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")

        # Return combined output for reasoning models
        if reasoning and not content:
            return f"[Reasoning: {reasoning}]\n[Content: empty - increase max_tokens]"
        elif reasoning:
            return f"[Reasoning: {reasoning}]\n{content}"
        else:
            return content

    def _test_streaming(self, payload):
        """Test streaming generation"""
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=120
        )

        text_parts = []
        reasoning_parts = []
        print("\n🤖 Streaming response:")

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0]["delta"]
                            # Handle reasoning content for reasoning models
                            if "reasoning_content" in delta:
                                reasoning_parts.append(delta["reasoning_content"])
                                print(f"[R: {delta['reasoning_content']}]", end="", flush=True)
                            if "content" in delta:
                                text_parts.append(delta["content"])
                                print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        print()  # New line after streaming

        # Combine reasoning and content
        reasoning = "".join(reasoning_parts)
        content = "".join(text_parts)

        if reasoning and not content:
            return f"[Reasoning: {reasoning}]\n[Content: empty - increase max_tokens]"
        elif reasoning:
            return f"[Reasoning: {reasoning}]\n{content}"
        else:
            return content

    def print_summary(self):
        """Print summary of all tests"""
        print(f"\n{'='*60}")
        print(f"SUMMARY: Generation Tests for {self.model_id}")
        print(f"{'='*60}")

        if not self.results:
            print("No tests completed")
            return

        print(f"\nCompleted {len(self.results)} tests:\n")

        for result in self.results:
            print(f"• {result['test_name']}:")
            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  Response length: {result['response_length']} chars")

        # Calculate averages
        avg_time = sum(r['elapsed_time'] for r in self.results) / len(self.results)
        avg_length = sum(r['response_length'] for r in self.results) / len(self.results)

        print(f"\nAverage time: {avg_time:.2f}s")
        print(f"Average response length: {avg_length:.0f} chars")


def run_comprehensive_tests(model_id):
    """Run comprehensive generation test suite"""
    tester = GenerationTester(model_id)

    print("="*60)
    print("🧪 Comprehensive Generation Test Suite")
    print("="*60)
    print(f"Testing model: {model_id}")
    print(f"API endpoint: {BASE_URL}")

    # Test 1: Short response
    tester.test_generation(
        "Short Response",
        "What is 2 + 2?",
        max_tokens=200
    )

    # Test 2: Medium response
    tester.test_generation(
        "Medium Response",
        "Explain what a neural network is in simple terms.",
        max_tokens=500
    )

    # Test 3: Creative task
    tester.test_generation(
        "Creative Task",
        "Write a short poem about programming.",
        max_tokens=400
    )

    # Test 4: Streaming test
    tester.test_generation(
        "Streaming Test",
        "List the first 10 prime numbers and explain what prime numbers are.",
        max_tokens=500,
        stream=True
    )

    # Test 5: Different temperature
    tester.test_generation(
        "Low Temperature (Precise)",
        "What is the capital of France?",
        max_tokens=200,
        temperature=0.3
    )

    # Test 6: High temperature
    tester.test_generation(
        "High Temperature (Creative)",
        "Write a creative opening for a sci-fi story.",
        max_tokens=500,
        temperature=1.5
    )

    # Print summary
    tester.print_summary()

    return tester.results


def compare_models(models):
    """Compare multiple models (for when you have both E2B and 26B)"""
    print("="*60)
    print("📊 Model Comparison Tests")
    print("="*60)

    all_results = {}

    for model in models:
        print(f"\nTesting {model}...")
        all_results[model] = run_comprehensive_tests(model)

    # Print comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)

    for model, results in all_results.items():
        if results:
            avg_time = sum(r['elapsed_time'] for r in results) / len(results)
            print(f"\n{model}:")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Tests completed: {len(results)}")


if __name__ == "__main__":
    # Default model (26B)
    model_id = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-4-26b-a4b"

    # Check server is running
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        print(f"✅ Server is running at {BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Server not running at {BASE_URL}")
        print("Please start the server first:")
        print("  ./start_server.sh")
        sys.exit(1)

    # Run tests
    run_comprehensive_tests(model_id)

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\nTo test a different model:")
    print("  python generation_tests.py <model_id>")
    print("\nMake sure the model is loaded in LM Studio before running tests.")