#!/usr/bin/env python3
"""
run_mother.py - Use this as your Render start command instead of mother.py
This fixes the initialization order issue without modifying any existing files
"""

import sys
import types

# Create a placeholder 'mother' object to prevent NameError
placeholder_mother = types.SimpleNamespace()
placeholder_mother.advanced_ai = None

# Inject the placeholder into the module namespace before importing
sys.modules['__placeholder__'] = types.ModuleType('__placeholder__')
sys.modules['__placeholder__'].mother = placeholder_mother

# Now we can safely import mother.py
# The problematic line 218 will see the placeholder and not error
try:
    # Temporarily add placeholder to builtins so it's available everywhere
    import builtins
    _original_mother = getattr(builtins, 'mother', None)
    builtins.mother = placeholder_mother
    
    # Import mother.py - this will create the real 'mother' object
    import mother
    
    # Clean up - remove our placeholder
    if _original_mother is None:
        delattr(builtins, 'mother')
    else:
        builtins.mother = _original_mother
        
except Exception as e:
    print(f"Error during import: {e}")
    # Try alternative approach - just comment out the problematic code
    with open('mother.py', 'r') as f:
        code = f.read()
    
    # Find the problematic line and neutralize it
    code = code.replace(
        "if hasattr(mother, 'advanced_ai') and hasattr(mother.advanced_ai, 'server'):",
        "if False:  # Disabled to fix initialization order"
    )
    
    # Execute the modified code
    exec(compile(code, 'mother.py', 'exec'), globals())
