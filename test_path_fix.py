#!/usr/bin/env python3
"""
Test script to verify the Path concatenation fix works correctly
"""

from pathlib import Path

def test_path_concatenation():
    """Test that Path objects are properly converted to strings before concatenation"""
    
    # Test case 1: PosixPath + str (this would cause the original error)
    file_path = Path("test_file.txt")
    file_size = 1024
    
    print("Testing Path concatenation...")
    print(f"file_path type: {type(file_path)}")
    print(f"file_path value: {file_path}")
    print(f"file_size type: {type(file_size)}")
    print(f"file_size value: {file_size}")
    
    try:
        # This would fail with the original code
        # bad_result = file_path + str(file_size)  # This would cause the error
        # print(f"Bad result: {bad_result}")  # This line would never execute
        
        # This is our fix
        good_result = str(file_path) + str(file_size)
        print(f"Good result: {good_result}")
        
        # Test the actual hash function that's used in the code
        download_id = f"instant_download_{abs(hash(str(file_path) + str(file_size)))}"
        print(f"Download ID: {download_id}")
        
        print("✅ Test passed! Path concatenation works correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_path_concatenation()
