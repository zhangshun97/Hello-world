# Something About Pytorch 0.4.0

### 0. Requirements

- Python -- 3.x
- Pytorch -- 0.4
- torchversion -- 0.2.1



## 1. Autograd

### 1.1 `requires_grad`

Every Tensor has a flag: `requires_grad` that allows for fine grained exclusion of subgraphs from gradient computation and can increase efficiency.

If there’s a single input to an operation that requires gradient, its output will also require gradient. Conversely, **only if all inputs don’t require gradient, the output also won’t require it**. Backward computation is never performed in the subgraphs, where all Tensors didn’t require gradients.

- Example of `requires_grad`

```python
>>> x = torch.randn(5, 5)  # requires_grad=False by default
>>> y = torch.randn(5, 5)  # requires_grad=False by default
>>> z = torch.randn((5, 5), requires_grad=True)
>>> a = x + y
>>> a.requires_grad
False
>>> b = a + z
>>> b.requires_grad
True
```

### 1.2 How to 'freeze' part of a model, like *fine-tuning*

- It's enough to set all the wanted part `requires_grad=False`
- The gradients will only be computed where **at least one of the inputs** of an operation that `requires_grad=True`
- Here is an example of fine-tuning a CNN

```python
model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
```

- **Note that** parameters of **newly constructed modules** have `requires_grad=True` by default! (Contrast to newly constructed Tensors.)

### 1.3 In-place operations with autograd (discouraged) 

Something like `x += y`.

You might never need to use them unless you’re operating under heavy memory pressure.

There are mainly two reasons that **in-place operations** are discouraged:

- In-place operations can potentially **overwrite** values required to compute gradients
- Every in-place operation actually requires the implementation to **rewrite** the computational graph

If you use in-place operations, please refer to [*in-place correctness check*](https://pytorch.org/docs/stable/notes/autograd.html#in-place-correctness-checks).



## 2. CUDA

### 2.1 Device

[`torch.cuda`](https://pytorch.org/docs/stable/cuda.html#module-torch.cuda) is used to set up and run CUDA operations.It keeps track of the currently selected GPU, and all CUDA tensors you allocate will by default be created on that device. The selected device can be changed with a [`torch.cuda.device`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device) context manager.Cross-GPU operations are not allowed by default. So, in general, the results will also be on the same device.

- Example

  ```python
  cuda = torch.device('cuda')     # Default CUDA device
  cuda0 = torch.device('cuda:0')
  cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)
  
  # create CUDA tensors
  x = torch.tensor([1., 2.], device=cuda0)
  # x.device is device(type='cuda', index=0)
  y = torch.tensor([1., 2.]).cuda()
  # y.device is device(type='cuda', index=0)
  
  with torch.cuda.device(1):
      # allocates a tensor on GPU 1
      a = torch.tensor([1., 2.], device=cuda)
  
      # transfers a tensor from CPU to GPU 1
      b = torch.tensor([1., 2.]).cuda()
      # a.device and b.device are device(type='cuda', index=1)
  
      # You can also use ``Tensor.to`` to transfer a tensor:
      b2 = torch.tensor([1., 2.]).to(device=cuda)
      # b.device and b2.device are device(type='cuda', index=1)
  
      c = a + b
      # c.device is device(type='cuda', index=1)
  
      z = x + y
      # z.device is device(type='cuda', index=0)
  
      # even within a context, you can specify the device
      # (or give a GPU index to the .cuda call)
      d = torch.randn(2, device=cuda2)
      e = torch.randn(2).to(cuda2)
      f = torch.randn(2).cuda(cuda2)
      # d.device, e.device, and f.device are all device(type='cuda', index=2)
  ```

- In general, the effect of asynchronous computation is **invisible** to the caller.

- You can **force synchronous** computation by setting environment variable `CUDA_LAUNCH_BLOCKING=1`, which may be useful when an error occurs on GPU.

*CUDA stream and synchronization refer to [this](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams).*

*Memory management please refer to [this](https://pytorch.org/docs/stable/notes/cuda.html#memory-management).*

### 2.2 Device-agnostic Code

#### 2.2.1 Single GPU

The first step is to determine whether the GPU should be used or not. A common pattern is to use Python’s `argparse` module to read in user arguments, and have a flag that can be used to disable CUDA, in combination with [`is_available()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.is_available). In the following, `args.device` results in a`torch.device` object that can be used to move tensors to CPU or CUDA. 

- The following code can be applied to any PyTorch scripts.

```python
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
```

- Now that we have `args.device`, we can use it to create a Tensor on the desired device.

```python
x = torch.empty((8, 42), device=args.device)
net = Network().to(device=args.device)
```

- In `dataloader`

```python
cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)
```

#### 2.2.2 Multiple GPUs

- use the `CUDA_VISIBLE_DEVICES` environment flag to **manage** which GPUs are available to PyTorch
- to manually control which GPU a tensor is created on, the best practice is to use a [`torch.cuda.device`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device) context manager
- A general scenario

```python
print("Outside device is 0")  # On device 0 (default in most scenarios)
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0
```

### 2.3 Create new Tensors in forward pass

```python
cuda = torch.device('cuda')
x_cpu = torch.empty(2)
x_gpu = torch.empty(2, device=cuda)
x_cpu_long = torch.empty(2, dtype=torch.int64)

y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
print(y_cpu)

    tensor([[ 0.3000,  0.3000],
            [ 0.3000,  0.3000],
            [ 0.3000,  0.3000]])

y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
print(y_gpu)

    tensor([[-5.0000, -5.0000],
            [-5.0000, -5.0000],
            [-5.0000, -5.0000]], device='cuda:0')

y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
print(y_cpu_long)

    tensor([[ 1,  2,  3]])
```

- If you want to create a tensor of the same type and size of another tensor, and fill it with either ones or zeros

```python
x_cpu = torch.empty(2, 3)
x_gpu = torch.empty(2, 3)  # this may not be a gpu array...

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)
```









## References

- https://pytorch.org/docs