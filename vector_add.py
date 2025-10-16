import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get this program's ID and calculate starting position
    pid = tl.program_id(axis=0)
    starting_index = pid * BLOCK_SIZE
    # Create offsets for all elements this program will handle
    offsets = starting_index + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle the last block (might have fewer elements)
    mask = offsets < n_elements
    # Load data from a and b
    a_values = tl.load(a + offsets, mask=mask)
    b_values = tl.load(b + offsets, mask=mask)
    # Perform the addition
    c_values = a_values + b_values
    # Store the result back to c
    tl.store(c + offsets, c_values, mask=mask)
   
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
