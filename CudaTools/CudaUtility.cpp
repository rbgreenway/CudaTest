#include "cudautility.h"


CudaUtil::CudaUtil()
{
	m_cudaDevice = -1;
}




CudaUtil::~CudaUtil()
{		
}





bool CudaUtil::GetCudaDeviceCount(int &count)
{	
	bool success = true;
	count = 0;
	int cnt = 0;
	cudaError_t result = cudaGetDeviceCount(&cnt);
	if (result == cudaSuccess)
	{
		count = cnt;
	}
	else
	{
		success = false;
		m_errMsg = cudaGetErrorString(result);
	}
	return success;
}


bool CudaUtil::GetComputeCapability(int &major, int &minor)
{
	bool success = true;

	cudaDeviceProp props;
	cudaError_t result = cudaGetDeviceProperties(&props, 0);
	if (result == cudaSuccess)
	{
		major = props.major;
		minor = props.minor;
	}
	else
	{
		major = 0;
		minor = 0;
		m_errMsg = cudaGetErrorString(result);
		success = false;
	}

	return success;
}

bool CudaUtil::GetDeviceName(std::string &name)
{
	bool success = true;

	cudaDeviceProp props;
	cudaError_t result = cudaGetDeviceProperties(&props, 0);
	if (result == cudaSuccess)
	{
		name = props.name;
	}
	else
	{
		name = "unknown";
		m_errMsg = cudaGetErrorString(result);
		success = false;
	}

	return success;
}

bool CudaUtil::GetDeviceMemory(uint64_t &totalMem, uint64_t &freeMem)
{
	bool success = true;

	size_t totmem = 0, freemem = 0;
	cudaError_t result = cudaMemGetInfo(&freemem, &totmem);
	if (result == cudaSuccess)
	{
		totalMem = totmem;
		freeMem = freemem;
	}
	else
	{
		totalMem = 0;
		freeMem = 0;
		m_errMsg = cudaGetErrorString(result);
		success = false;
	}

	return success;
}


std::string CudaUtil::GetLastErrorMessage()
{
	return m_errMsg;
}

//
//
//std::string CudaUtil::GetCudaErrorMessage(CUresult cudaResult)
//{
//	char msg[2048];
//	const char* pmsg = &msg[0];
//	const char** ppmsg = &pmsg;
//	CUresult result = cuGetErrorString(cudaResult, ppmsg);
//
//	if (result != CUDA_SUCCESS)
//	{
//		m_errMsg = "Failed to retrieve error message for CUresult = " + std::to_string((int)cudaResult);
//	}
//	else
//	{
//		std::string errMsg(pmsg);
//		m_errMsg = errMsg;
//	}
//
//	return m_errMsg;
//}
//
//
//std::string CudaUtil::GetCudaErrorDescription(CUresult result)
//{
//	// these error descriptions are taken from cuda.h
//
//	std::string errMsg;
//
//	switch (result)
//	{
//
//	case CUDA_SUCCESS:
//		errMsg = "No Errors";
//		break;
//
//	case CUDA_ERROR_INVALID_VALUE:
//		errMsg = "One or more of the parameters passed to the API call is not within an acceptable range of values.";
//		break;
//
//	case CUDA_ERROR_OUT_OF_MEMORY:
//		errMsg = "The API call failed because it was unable to allocate enough memory to perform the requested operation.";
//		break;
//
//	case CUDA_ERROR_NOT_INITIALIZED:
//		errMsg = "The CUDA driver has not been initialized with ::cuInit() or that initialization has failed.";
//		break;
//
//	case CUDA_ERROR_DEINITIALIZED:
//		errMsg = "The CUDA driver is in the process of shutting down.";
//		break;
//
//	case CUDA_ERROR_PROFILER_DISABLED:
//		errMsg = "Profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler.";
//		break;
//
//	case CUDA_ERROR_NO_DEVICE:
//		errMsg = "No CUDA-capable devices were detected by the installed CUDA driver.";
//		break;
//
//	case CUDA_ERROR_INVALID_DEVICE:
//		errMsg = "The device ordinal supplied by the user does not correspond to a valid CUDA device.";
//		break;
//
//	case CUDA_ERROR_INVALID_IMAGE:
//		errMsg = "the device kernel image is invalid. This can also indicate an invalid CUDA module.";
//		break;
//
//	case CUDA_ERROR_INVALID_CONTEXT:
//		errMsg = "This most frequently indicates that there is no context bound to the\ncurrent thread. This can also be returned if the context passed to an\nAPI call is not a valid handle (such as a context that has had\n::cuCtxDestroy() invoked on it). This can also be returned if a user\nmixes different API versions (i.e. 3010 context with 3020 API calls).\nSee ::cuCtxGetApiVersion() for more details.";
//		break;
//
//	case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
//		errMsg = "The context being supplied as a parameter to the API call was already the active context.";
//		break;
//
//	case CUDA_ERROR_MAP_FAILED:
//		errMsg = "A map or register operation has failed.";
//		break;
//
//	case CUDA_ERROR_UNMAP_FAILED:
//		errMsg = "A unmap or register operation has failed.";
//		break;
//
//	case CUDA_ERROR_ARRAY_IS_MAPPED:
//		errMsg = "The specified array is currently mapped and thus cannot be destroyed.";
//		break;
//
//	case CUDA_ERROR_ALREADY_MAPPED:
//		errMsg = "The resource is already mapped.";
//		break;
//
//	case CUDA_ERROR_NO_BINARY_FOR_GPU:
//		errMsg = "There is no kernel image available that is suitable\n for the device. This can occur when a user specifies code generation\noptions for a particular CUDA source file that do not include the\ncorresponding device configuration.";
//		break;
//
//	case CUDA_ERROR_ALREADY_ACQUIRED:
//		errMsg = "A resource has already been acquired.";
//		break;
//
//	case CUDA_ERROR_NOT_MAPPED:
//		errMsg = "A resource is not mapped.";
//		break;
//
//	case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
//		errMsg = "A mapped resource is not available for access as an array.";
//		break;
//
//	case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
//		errMsg = "A mapped resource is not available for access as a pointer.";
//		break;
//
//	case CUDA_ERROR_ECC_UNCORRECTABLE:
//		errMsg = "An uncorrectable ECC error was detected during execution.";
//		break;
//
//	case CUDA_ERROR_UNSUPPORTED_LIMIT:
//		errMsg = "The ::CUlimit passed to the API call is not supported by the active device.";
//		break;
//
//	case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
//		errMsg = "The ::CUcontext passed to the API call can only be bound\nto a single CPU thread at a time but is already bound\nto a CPU thread.";
//		break;
//
//	case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
//		errMsg = "Peer access is not supported across the given devices.";
//		break;
//
//	case CUDA_ERROR_INVALID_PTX:
//		errMsg = "A PTX JIT compilation failed.";
//		break;
//
//	case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
//		errMsg = "An error with OpenGL or DirectX context.";
//		break;
//
//	case CUDA_ERROR_INVALID_SOURCE:
//		errMsg = "The device kernel source is invalid.";
//		break;
//
//	case CUDA_ERROR_FILE_NOT_FOUND:
//		errMsg = "The file specified was not found.";
//		break;
//
//	case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
//		errMsg = "A link to a shared object failed to resolve.";
//		break;
//
//	case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
//		errMsg = "Initialization of a shared object failed.";
//		break;
//
//	case CUDA_ERROR_OPERATING_SYSTEM:
//		errMsg = "An OS call failed.";
//		break;
//
//	case CUDA_ERROR_INVALID_HANDLE:
//		errMsg = "A resource handle passed to the API call was not valid.\nResource handles are opaque types like ::CUstream and ::CUevent.";
//		break;
//
//	case CUDA_ERROR_NOT_FOUND:
//		errMsg = "A named symbol was not found. Examples of symbols are global/constant\nvariable names, texture names, and surface names.";
//		break;
//
//	case CUDA_ERROR_NOT_READY:
//		errMsg = "Asynchronous operations issued previously have not completed yet.\nThis result is not actually an error, but must be indicated\ndifferently than ::CUDA_SUCCESS (which indicates completion). Calls that\nmay return this value include ::cuEventQuery() and ::cuStreamQuery().";
//		break;
//
//	case CUDA_ERROR_ILLEGAL_ADDRESS:
//		errMsg = "While executing a kernel, the device encountered a\nload or store instruction on an invalid memory address.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
//		errMsg = "This indicates that a launch did not occur because it did not have\nappropriate resources. This error usually indicates that the user has\nattempted to pass too many arguments to the device kernel, or the\nkernel launch specifies too many threads for the kernel's register\ncount. Passing arguments of the wrong size (i.e. a 64-bit pointer\nwhen a 32-bit int is expected) is equivalent to passing too many\narguments and can also result in this error.";
//		break;
//
//	case CUDA_ERROR_LAUNCH_TIMEOUT:
//		errMsg = "This indicates that the device kernel took too long to execute. This can\nonly occur if timeouts are enabled - see the device attribute\n::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The\ncontext cannot be used (and must be destroyed similar to\n::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from\nthis context are invalid and must be reconstructed if the program is to\ncontinue using CUDA.";
//		break;
//
//	case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
//		errMsg = "A kernel launch that uses an incompatible texturing mode.";
//		break;
//
//	case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
//		errMsg = "A call to ::cuCtxEnablePeerAccess() is trying to re-enable peer\naccess to a context which has already had peer access to it enabled.";
//		break;
//
//	case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
//		errMsg = "::cuCtxDisablePeerAccess() is trying to disable peer access which has not been\nenabled yet via ::cuCtxEnablePeerAccess().";
//		break;
//
//	case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
//		errMsg = "The primary context for the specified device has already been initialized.";
//		break;
//
//	case CUDA_ERROR_CONTEXT_IS_DESTROYED:
//		errMsg = "The context current to the calling thread has been destroyed using\n::cuCtxDestroy, or is a primary context which has not yet been initialized.";
//		break;
//
//	case CUDA_ERROR_ASSERT:
//		errMsg = "A device-side assert triggered during kernel execution. The context\ncannot be used anymore, and must be destroyed. All existing device\nmemory allocations from this context are invalid and must be\nreconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_TOO_MANY_PEERS:
//		errMsg = "The hardware resources required to enable peer access have been\nexhausted for one or more of the devices passed to ::cuCtxEnablePeerAccess().";
//		break;
//
//	case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
//		errMsg = "The memory range passed to ::cuMemHostRegister() has already been registered.";
//		break;
//
//	case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
//		errMsg = "The pointer passed to ::cuMemHostUnregister() does not correspond to any currently registered memory region.";
//		break;
//
//	case CUDA_ERROR_HARDWARE_STACK_ERROR:
//		errMsg = "While executing a kernel, the device encountered a stack error.\nThis can be due to stack corruption or exceeding the stack size limit.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_ILLEGAL_INSTRUCTION:
//		errMsg = "While executing a kernel, the device encountered an illegal instruction.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_MISALIGNED_ADDRESS:
//		errMsg = "While executing a kernel, the device encountered a load or store instruction\non a memory address which is not aligned.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_INVALID_ADDRESS_SPACE:
//		errMsg = "While executing a kernel, the device encountered an instruction\nwhich can only operate on memory locations in certain address spaces\n(global, shared, or local), but was supplied a memory address not\nbelonging to an allowed address space.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_INVALID_PC:
//		errMsg = "While executing a kernel, the device program counter wrapped its address space.\nThe context cannot be used, so it must be destroyed (and a new one should be created).\nAll existing device memory allocations from this context are invalid\nand must be reconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_LAUNCH_FAILED:
//		errMsg = "An exception occurred on the device while executing a kernel. Common\ncauses include dereferencing an invalid device pointer and accessing\nout of bounds shared memory. The context cannot be used, so it must\nbe destroyed (and a new one should be created). All existing device\nmemory allocations from this context are invalid and must be\nreconstructed if the program is to continue using CUDA.";
//		break;
//
//	case CUDA_ERROR_NOT_PERMITTED:
//		errMsg = "The attempted operation is not permitted.";
//		break;
//
//	case CUDA_ERROR_NOT_SUPPORTED:
//		errMsg = "The attempted operation is not supported on the current system or device.";
//		break;
//
//	case CUDA_ERROR_UNKNOWN:
//		errMsg = "An unknown internal error has occurred.";
//		break;
//
//	default:
//		errMsg = "An unknown CUDA error.";
//		break;
//	}
//
//	return errMsg;
//}
