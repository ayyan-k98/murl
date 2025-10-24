#!/usr/bin/env python3
"""
Verify Device Mismatch Fix

This script checks if the device mismatch fix is present in the codebase.
"""

import os
import sys

def check_file(filepath, expected_patterns):
    """Check if file contains expected patterns."""
    if not os.path.exists(filepath):
        return False, f"File not found: {filepath}"

    with open(filepath, 'r') as f:
        content = f.read()

    missing_patterns = []
    for pattern in expected_patterns:
        if pattern not in content:
            missing_patterns.append(pattern)

    if missing_patterns:
        return False, f"Missing patterns in {filepath}: {missing_patterns}"

    return True, f"✅ {filepath} has device fix"

def main():
    print("=" * 80)
    print("VERIFYING DEVICE MISMATCH FIX")
    print("=" * 80)
    print()

    # Patterns to check for in agent.py
    agent_patterns = [
        "state_graphs_cpu",
        "s.to('cpu') if s.x.device.type != 'cpu' else s",
        "Batch.from_data_list(state_graphs_cpu).to(self.device)"
    ]

    # Patterns to check for in agent_enhanced.py
    agent_enhanced_patterns = [
        "state_graphs_cpu",
        "s.to('cpu') if s.x.device.type != 'cpu' else s",
        "Batch.from_data_list(state_graphs_cpu).to(self.device)"
    ]

    all_passed = True

    # Check agent.py
    success, message = check_file('agent.py', agent_patterns)
    print(message)
    if not success:
        all_passed = False

    # Check agent_enhanced.py
    success, message = check_file('agent_enhanced.py', agent_enhanced_patterns)
    print(message)
    if not success:
        all_passed = False

    print()
    print("=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Device mismatch fix is present!")
        print()
        print("The fix ensures all graphs are moved to CPU before batching,")
        print("preventing the RuntimeError on CUDA devices.")
    else:
        print("❌ SOME CHECKS FAILED - Device fix may be missing!")
        print()
        print("Please ensure you have the latest code from:")
        print("  Branch: claude/analyze-training-process-011CURYb8BTGox4XojBk3cxf")
        print("  Commit: 83021f3")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
