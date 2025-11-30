import struct
import json
import sys

filename = "model.safetensors"

with open(filename, "rb") as f:
    # Read the header size (first 8 bytes)
    header_size_bytes = f.read(8)
    if len(header_size_bytes) != 8:
        print("File too short")
        sys.exit(1)
    
    header_size = struct.unpack("<Q", header_size_bytes)[0]
    print(f"Header size: {header_size}")
    
    # Read the header
    header_json = f.read(header_size)
    header = json.loads(header_json)
    
    # Analyze data types
    dtypes = {}
    total_elements = 0
    
    print("\nTensor analysis:")
    for key, value in header.items():
        if key == "__metadata__":
            continue
        
        dtype = value['dtype']
        shape = value['shape']
        
        # Calculate number of elements
        num_elements = 1
        for dim in shape:
            num_elements *= dim
            
        dtypes[dtype] = dtypes.get(dtype, 0) + num_elements
        total_elements += num_elements
        
    print("\nData Type Distribution (by number of elements):")
    for dtype, count in dtypes.items():
        print(f"{dtype}: {count} elements ({count/total_elements*100:.2f}%)")

