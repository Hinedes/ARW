import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

# 1. Find and print the location of the source file
print(f"Source file: {inspect.getfile(Gemma4TextRotaryEmbedding)}")

# 2. Print the full source code of the class (be careful, this can be long)
# print(inspect.getsource(Gemma4TextRotaryEmbedding))

# 3. Inspect the signature of the forward method
try:
    sig = inspect.signature(Gemma4TextRotaryEmbedding.forward)
    print("\nForward method signature:")
    for name, param in sig.parameters.items():
        print(f"  - {name}: {param.default if param.default is not inspect.Parameter.empty else 'No default'}")
except Exception as e:
    print(f"Could not get signature: {e}")

# 4. As a final fallback, check what attributes exist after initialization
# (This might require instantiating, which we'll do if needed)