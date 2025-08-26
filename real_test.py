import neurolea

print("🔍 TESTING REAL NEUROLEA CAPABILITIES")
print("=" * 50)

try:
    framework = neurolea.UltimateAIFramework()
    
    # Test 1: Can it actually process text?
    print("\n1️⃣ Testing Text Processing...")
    test_data = ["Hello world", "AI is amazing", "Test sentence"]
    
    try:
        framework.initialize_all_components(test_data)
        print("   ✅ Text processing: SUCCESS")
    except Exception as e:
        print(f"   ❌ Text processing: FAILED - {e}")
    
    # Test 2: Can it generate text?
    print("\n2️⃣ Testing Text Generation...")
    try:
        if hasattr(framework, 'hybrid_model') and framework.hybrid_model:
            result = framework.hybrid_model.generate([1, 2, 3], max_new_tokens=5)
            print(f"   ✅ Generation: SUCCESS - {result}")
        else:
            print("   ❌ Generation: No model available")
    except Exception as e:
        print(f"   ❌ Generation: FAILED - {e}")
    
    # Test 3: What's actually working?
    print(f"\n📊 Framework Status:")
    print(f"   Capabilities: {framework.capabilities}")
    print(f"   Has tokenizer: {hasattr(framework, 'tokenizer')}")
    print(f"   Has model: {hasattr(framework, 'hybrid_model')}")
    print(f"   Has code_optimizer: {hasattr(framework, 'code_optimizer')}")
    print(f"   Has multimodal_model: {hasattr(framework, 'multimodal_model')}")
    
    # Test 4: Try a simple capability
    print("\n3️⃣ Testing Simple Generation...")
    try:
        # Test if tokenizer works
        if hasattr(framework, 'tokenizer') and framework.tokenizer:
            tokens = framework.tokenizer.encode("Hello AI")
            print(f"   ✅ Tokenization: SUCCESS - {tokens}")
        else:
            print("   ❌ Tokenization: No tokenizer")
    except Exception as e:
        print(f"   ❌ Tokenization: FAILED - {e}")
    
    print("\n🎯 VERDICT: Framework analysis complete")
    
except Exception as e:
    print(f"❌ MAJOR ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 Real test completed!")
