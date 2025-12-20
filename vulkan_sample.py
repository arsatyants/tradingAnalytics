#!/usr/bin/env python3
"""
Vulkan Compute Sample in Python
Demonstrates basic Vulkan usage for GPU computation
"""

import vulkan as vk
import numpy as np

def create_instance():
    """Create Vulkan instance"""
    app_info = vk.VkApplicationInfo(
        pApplicationName="Vulkan Compute Sample",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0
    )
    
    create_info = vk.VkInstanceCreateInfo(
        pApplicationInfo=app_info,
        enabledLayerCount=0,
        enabledExtensionCount=0
    )
    
    instance = vk.vkCreateInstance(create_info, None)
    return instance

def find_compute_device(instance):
    """Find a GPU device that supports compute operations"""
    physical_devices = vk.vkEnumeratePhysicalDevices(instance)
    
    if not physical_devices:
        raise RuntimeError("No Vulkan devices found")
    
    for device in physical_devices:
        props = vk.vkGetPhysicalDeviceProperties(device)
        print(f"Found device: {props.deviceName}")
        
        # Check for compute queue family
        queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(device)
        for i, family in enumerate(queue_families):
            if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT:
                print(f"  -> Compute queue family {i}")
                return device, i
    
    raise RuntimeError("No compute-capable device found")

def create_logical_device(physical_device, queue_family_index):
    """Create logical device with compute queue"""
    queue_create_info = vk.VkDeviceQueueCreateInfo(
        queueFamilyIndex=queue_family_index,
        queueCount=1,
        pQueuePriorities=[1.0]
    )
    
    device_create_info = vk.VkDeviceCreateInfo(
        queueCreateInfoCount=1,
        pQueueCreateInfos=[queue_create_info],
        enabledLayerCount=0,
        enabledExtensionCount=0
    )
    
    device = vk.vkCreateDevice(physical_device, device_create_info, None)
    queue = vk.vkGetDeviceQueue(device, queue_family_index, 0)
    
    return device, queue

def create_buffer(device, physical_device, size, usage):
    """Create Vulkan buffer"""
    buffer_info = vk.VkBufferCreateInfo(
        size=size,
        usage=usage,
        sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE
    )
    
    buffer = vk.vkCreateBuffer(device, buffer_info, None)
    
    # Get memory requirements
    mem_requirements = vk.vkGetBufferMemoryRequirements(device, buffer)
    
    # Find suitable memory type
    mem_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical_device)
    memory_type_index = None
    
    for i in range(mem_properties.memoryTypeCount):
        if (mem_requirements.memoryTypeBits & (1 << i)) and \
           (mem_properties.memoryTypes[i].propertyFlags & 
            (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)):
            memory_type_index = i
            break
    
    if memory_type_index is None:
        raise RuntimeError("Failed to find suitable memory type")
    
    # Allocate memory
    alloc_info = vk.VkMemoryAllocateInfo(
        allocationSize=mem_requirements.size,
        memoryTypeIndex=memory_type_index
    )
    
    memory = vk.vkAllocateMemory(device, alloc_info, None)
    vk.vkBindBufferMemory(device, buffer, memory, 0)
    
    return buffer, memory

def main():
    """Main Vulkan compute example"""
    print("=" * 60)
    print("VULKAN COMPUTE SAMPLE")
    print("=" * 60)
    
    # Initialize Vulkan
    print("\n1. Creating Vulkan instance...")
    instance = create_instance()
    
    # Find compute device
    print("\n2. Finding compute-capable device...")
    physical_device, queue_family = find_compute_device(instance)
    
    # Get device properties
    props = vk.vkGetPhysicalDeviceProperties(physical_device)
    print(f"\nSelected device: {props.deviceName}")
    print(f"  API Version: {vk.VK_VERSION_MAJOR(props.apiVersion)}.{vk.VK_VERSION_MINOR(props.apiVersion)}")
    print(f"  Driver Version: {props.driverVersion}")
    print(f"  Vendor ID: {hex(props.vendorID)}")
    print(f"  Device ID: {hex(props.deviceID)}")
    
    # Create logical device
    print("\n3. Creating logical device and queue...")
    device, queue = create_logical_device(physical_device, queue_family)
    
    # Create buffers for computation
    print("\n4. Creating buffers...")
    buffer_size = 1024 * 4  # 1024 floats = 4096 bytes
    
    input_buffer, input_memory = create_buffer(
        device, physical_device, buffer_size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
    )
    
    output_buffer, output_memory = create_buffer(
        device, physical_device, buffer_size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
    )
    
    print(f"  Input buffer created: {buffer_size} bytes")
    print(f"  Output buffer created: {buffer_size} bytes")
    
    # Write data to input buffer
    print("\n5. Creating test data...")
    data = np.arange(1024, dtype=np.float32)
    print(f"  Created {len(data)} floats in CPU memory")
    print(f"  Note: Memory mapping/writing skipped due to API complexity")
    print(f"  In production, you would use vkMapMemory + memcpy or staging buffers")
    
    # In a real application, you would:
    # 1. Map memory: data_ptr = vk.vkMapMemory(device, input_memory, 0, buffer_size, 0)
    # 2. Copy data using proper FFI buffer protocol
    # 3. Unmap memory: vk.vkUnmapMemory(device, input_memory)
    # Or use staging buffers with vkCmdCopyBuffer for better performance
    
    # Note: To actually compute, you'd need to create compute pipeline with shaders
    # This example shows the basic setup
    
    print("\n6. Cleanup...")
    vk.vkDestroyBuffer(device, input_buffer, None)
    vk.vkDestroyBuffer(device, output_buffer, None)
    vk.vkFreeMemory(device, input_memory, None)
    vk.vkFreeMemory(device, output_memory, None)
    vk.vkDestroyDevice(device, None)
    vk.vkDestroyInstance(instance, None)
    
    print("\n" + "=" * 60)
    print("✓ Vulkan initialization and buffer operations successful!")
    print("=" * 60)
    print("\nTo run actual compute shaders, you would need to:")
    print("  1. Create compute shader (SPIR-V binary)")
    print("  2. Create descriptor set layout")
    print("  3. Create pipeline layout")
    print("  4. Create compute pipeline")
    print("  5. Create command pool and buffers")
    print("  6. Record and submit commands")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure vulkan package is installed:")
        print("  pip install vulkan")
        print("\nAnd Vulkan drivers are available on your system")
