# Fused Kernel Library (FKL)

The Fused Kernel Library is a C++17 + CUDA implementation of a methodology (publication pending) that allows to define a set of GPU code elements that need to be executed inside a CUDA kernel, in the same order that they are expressed.

It automatically implements Vertical and Horizontal fusion, and also implements two new Fusion techniques, Backwards Vertical Fusion (akin to OpenCV Filters, but with an standard generic API), and Divergent Horizontal Fusion.

## Fusion and inclusion

The way the FusedKernel library (FKL) is implemented, allows not only to use the already implemented Operations and data types like Ptr2D or Tensor, but also the fusion can be performend using any code that conforms to the FusedKernel interface (the InstantiableOperation structs and the OperationTypes). The operations in FKL can use any data type that the user wants to use, basic types, structs, tuples (we implemented fk::Tuple to be used on GPU code, along with fk::apply and other utilites).

This was done in purpose to make it easier to join efforts with other libraries that already exist and are also OpenSource and want to take advantage of the FusedKernelLibrary methodology.

Additionally, this flexibility helps companies with closed-source frameworks to apply the FKL methodology and even make use of the library, combining it with their own code and data types.
### Horizontal Fusion

This fusion technique is widely known and used. It is based on the idea of processing several data planes in parallel, with the same CUDA kernel. For that, we use the blockIdx.z, to distinguish between thread planes and data planes.

This is usually very beneficial when each plane is very small, and the resulting 2D Grid is not taking advantage of the GPU memory bandwidth.

We also support what we call Divergent Horizontal Fusion. This variant allows to execute different kernels that can be executed in parallel. Each "kernel" can use one or more z planes of the grid, so each kernel can do Horizontal Fusion. This technique allows to exploit the possibility of using diferent components in the SM's in parallel, improving the overall performance.

### Vertical Fusion (Generic Vertical Fusion)

Vertical Fusion is usually limited to having a kernel that is configurable up to a certain level, or there is a list of pre-compiled fused kernels to choose from. In our case, we are abstrating away the thread behavior from the actual functionality, and allowing to fuse almost every kernel possible, without having to rewrite neither the thread handling, nor the functionality. You only have to combine code in the different ways that the code can be combined. We call this Generic Vertical Fusion.

For Memory Bound kernels, vertical fusion is bringing most of the performance improvements possible, since adding more functions to the kernel will not increase the execution time, up to a limit where the kernel becomes Compute Bound.

Not only that, but thanks to the way the code is written, the nvcc compiler will treat the consecutive operations as if you where writting the code in one line, adding all sorts of optimizations. This can be seen by compiling the code in Release mode, or in Debug mode. The performance difference is abismal.

### Backwards Vertical Fusion (read and compute, only what you need)

This is an optimization that can already be used with the current code, but will be refined and further increase the use cases when adding more Operations.

The idea, is aplicable for situations where you have a big plane, from which you will only use a subset of the data. If you need to transform that plane into something different before before operating on the subset, you can use Backwards Vertical Fusion in order to have a single kernel, that will read only what it needs, and apply to it all the operations needed.

For example, let's assume that you receive an image in YUV420_NV12 format, and you need to crop a region of this image, then convert the pixels to RGB, then resize the crop, normalize it to floating point values from 0 to 1, and store the resulting image in RGB planar format. Usually, this would lead to many kernels, one after the other. The first kernel that converts to RGB, will convert the full image, and write the result to memory. Instead, with the Fused Kernel library, it is possible to create a Fused Kernel in a few lines, that will only read the YUV data for the pixels required by the interpolation process, in the resize of the crop. All the steps will be performed using GPU registers, until the last step where we will finally write into GPU ram memory.

This is way faster to program than the conventional way of programming CUDA.

### Divergent Horizontal Fusion

This novel type of Horizontal Fusion, allows to Horizontally Fuse kernels that are completelly different, and read the same or different data, but write the results in different memory regions.

This has been tested before, by creating special compilers that generate the assembly code, and the performance benefits have been already reported. The novelty in our approach is that we do not require a different compiler. We do this by leveraging the C++17 capabilities found in nvcc.

## Closed source friendly

A company that has it's own CUDA kernels, and wants to start fusing them along with operations present in this library, can do so by shaping their kernels into a conformant FusedKernel Operation, that can be passed as a template parameter of one of the FKL InstantiableOperation structs.

With this strategy, they don't need to share any of their code. They just need to make their kernels fusionable.
