//! Types for error handling
//!
//! # Error handling in CUDA:
//!
//! cust uses the [`CudaError`](enum.CudaError.html) enum to represent the errors returned by
//! the CUDA API. It is important to note that nearly every function in CUDA (and therefore
//! cust) can fail. Even those functions which have no normal failure conditions can return
//! errors related to previous asynchronous launches.

use crate::sys::{self as cuda, cudaError_enum};
use std::error::Error;
use std::ffi::CStr;
use std::fmt;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::result::Result;

/// Error enum which represents all the potential errors returned by the CUDA driver API.
#[repr(u32)]
#[allow(missing_docs)]
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaError {
    // CUDA errors
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    StubLibrary = 34,
    DeviceUnavailable = 46,
    NoDevice = 100,
    InvalidDevice = 101,
    DeviceNotLicensed = 102,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    EccUncorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidPtx = 218,
    InvalidGraphicsContext = 219,
    NvlinkUncorrectable = 220,
    JitCompilerNotFound = 221,
    UnsupportedPtxVersion = 222,
    JitCompilationDisabled = 223,
    UnsupportedExecAffinity = 224,
    UnsupportedDevSideSync = 225,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystemError = 304,
    InvalidHandle = 400,
    IllegalState = 401,
    LossyQuery = 402,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    PrimaryContextActive = 708,
    ContextIsDestroyed = 709,
    AssertError = 710,
    TooManyPeers = 711,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    HardwareStackError = 714,
    IllegalInstruction = 715,
    MisalignedAddress = 716,
    InvalidAddressSpace = 717,
    InvalidProgramCounter = 718,
    LaunchFailed = 719,
    CooperativeLaunchTooLarge = 720,
    NotPermitted = 800,
    NotSupported = 801,
    SystemNotReady = 802,
    SystemDriverMismatch = 803,
    CompatNotSupportedOnDevice = 804,
    MpsConnectionFailed = 805,
    MpsRpcFailed = 806,
    MpsServerNotReady = 807,
    MpsMaxClientsReached = 808,
    MpsMaxConnectionsReached = 809,
    MpsClientTerminated = 810,
    CdpNotSupported = 811,
    CdpVersionMismatch = 812,
    StreamCaptureUnsupported = 900,
    StreamCaptureInvalid = 901,
    StreamCaptureMerge = 902,
    StreamCaptureUnmatched = 903,
    StreamCaptureUnjoined = 904,
    StreamCaptureIsolated = 905,
    StreamCaptureImplicit = 906,
    CapturedEvent = 907,
    StreamCaptureWrongThread = 908,
    Timeout = 909,
    GraphExecUpdateFailure = 910,
    ExternalDevice = 911,
    InvalidClusterSize = 912,
    FunctionNotLoaded = 913,
    InvalidResourceType = 914,
    InvalidResourceConfiguration = 915,
    UnknownError = 999,

    // cust errors
    InvalidMemoryAllocation = 100_100,
    OptixError = 100_101,
}
impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            CudaError::InvalidMemoryAllocation => write!(f, "Invalid memory allocation"),
            CudaError::OptixError => write!(f, "OptiX error"),
            other if (other as u32) <= 999 => {
                let value = other as u32;
                let mut ptr: *const c_char = ptr::null();
                unsafe {
                    cuda::cuGetErrorString(mem::transmute(value), &mut ptr as *mut *const c_char)
                        .to_result()
                        .map_err(|_| fmt::Error)?;
                    let cstr = CStr::from_ptr(ptr);
                    write!(f, "{:?}", cstr)
                }
            }
            // This shouldn't happen
            _ => write!(f, "Unknown error"),
        }
    }
}
impl Error for CudaError {}

/// Result type for most CUDA functions.
pub type CudaResult<T> = Result<T, CudaError>;

/// Special result type for `drop` functions which includes the un-dropped value with the error.
pub type DropResult<T> = Result<(), (CudaError, T)>;

pub(crate) trait ToResult {
    fn to_result(self) -> CudaResult<()>;
}
impl ToResult for cudaError_enum {
    fn to_result(self) -> CudaResult<()> {
        match self {
            cudaError_enum::CUDA_SUCCESS => Ok(()),
            cudaError_enum::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            cudaError_enum::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            cudaError_enum::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            cudaError_enum::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            cudaError_enum::CUDA_ERROR_PROFILER_NOT_INITIALIZED => {
                Err(CudaError::ProfilerNotInitialized)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STARTED => {
                Err(CudaError::ProfilerAlreadyStarted)
            }
            cudaError_enum::CUDA_ERROR_PROFILER_ALREADY_STOPPED => {
                Err(CudaError::ProfilerAlreadyStopped)
            }
            cudaError_enum::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            cudaError_enum::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            cudaError_enum::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            cudaError_enum::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => {
                Err(CudaError::ContextAlreadyCurrent)
            }
            cudaError_enum::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            cudaError_enum::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            cudaError_enum::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            cudaError_enum::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            cudaError_enum::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            cudaError_enum::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            cudaError_enum::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            cudaError_enum::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            cudaError_enum::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => {
                Err(CudaError::ContextAlreadyInUse)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => {
                Err(CudaError::PeerAccessUnsupported)
            }
            cudaError_enum::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            cudaError_enum::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => {
                Err(CudaError::InvalidGraphicsContext)
            }
            cudaError_enum::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            cudaError_enum::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSource),
            cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            cudaError_enum::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => {
                Err(CudaError::SharedObjectInitFailed)
            }
            cudaError_enum::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            cudaError_enum::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            cudaError_enum::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            cudaError_enum::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            cudaError_enum::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            cudaError_enum::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => {
                Err(CudaError::LaunchOutOfResources)
            }
            cudaError_enum::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            cudaError_enum::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CudaError::PeerAccessAlreadyEnabled)
            }
            cudaError_enum::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => {
                Err(CudaError::PeerAccessNotEnabled)
            }
            cudaError_enum::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => {
                Err(CudaError::PrimaryContextActive)
            }
            cudaError_enum::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            cudaError_enum::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            cudaError_enum::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            cudaError_enum::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => {
                Err(CudaError::HostMemoryNotRegistered)
            }
            cudaError_enum::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            cudaError_enum::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            cudaError_enum::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            cudaError_enum::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            cudaError_enum::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            cudaError_enum::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            cudaError_enum::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            cudaError_enum::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            cudaError_enum::CUDA_ERROR_STUB_LIBRARY => Err(CudaError::StubLibrary),
            cudaError_enum::CUDA_ERROR_DEVICE_UNAVAILABLE => Err(CudaError::DeviceUnavailable),
            cudaError_enum::CUDA_ERROR_DEVICE_NOT_LICENSED => Err(CudaError::DeviceNotLicensed),
            cudaError_enum::CUDA_ERROR_JIT_COMPILER_NOT_FOUND => Err(CudaError::JitCompilerNotFound),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_PTX_VERSION => Err(CudaError::UnsupportedPtxVersion),
            cudaError_enum::CUDA_ERROR_JIT_COMPILATION_DISABLED => Err(CudaError::JitCompilationDisabled),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => Err(CudaError::UnsupportedExecAffinity),
            cudaError_enum::CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC => Err(CudaError::UnsupportedDevSideSync),
            cudaError_enum::CUDA_ERROR_ILLEGAL_STATE => Err(CudaError::IllegalState),
            cudaError_enum::CUDA_ERROR_LOSSY_QUERY => Err(CudaError::LossyQuery),
            cudaError_enum::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE => Err(CudaError::CooperativeLaunchTooLarge),
            cudaError_enum::CUDA_ERROR_SYSTEM_NOT_READY => Err(CudaError::SystemNotReady),
            cudaError_enum::CUDA_ERROR_SYSTEM_DRIVER_MISMATCH => Err(CudaError::SystemDriverMismatch),
            cudaError_enum::CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE => Err(CudaError::CompatNotSupportedOnDevice),
            cudaError_enum::CUDA_ERROR_MPS_CONNECTION_FAILED => Err(CudaError::MpsConnectionFailed),
            cudaError_enum::CUDA_ERROR_MPS_RPC_FAILURE => Err(CudaError::MpsRpcFailed),
            cudaError_enum::CUDA_ERROR_MPS_SERVER_NOT_READY => Err(CudaError::MpsServerNotReady),
            cudaError_enum::CUDA_ERROR_MPS_MAX_CLIENTS_REACHED => Err(CudaError::MpsMaxClientsReached),
            cudaError_enum::CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED => Err(CudaError::MpsMaxConnectionsReached),
            cudaError_enum::CUDA_ERROR_MPS_CLIENT_TERMINATED => Err(CudaError::MpsClientTerminated),
            cudaError_enum::CUDA_ERROR_CDP_NOT_SUPPORTED => Err(CudaError::CdpNotSupported),
            cudaError_enum::CUDA_ERROR_CDP_VERSION_MISMATCH => Err(CudaError::CdpVersionMismatch),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED => Err(CudaError::StreamCaptureUnsupported),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_INVALIDATED => Err(CudaError::StreamCaptureInvalid),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_MERGE => Err(CudaError::StreamCaptureMerge),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNMATCHED => Err(CudaError::StreamCaptureUnmatched),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_UNJOINED => Err(CudaError::StreamCaptureUnjoined),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_ISOLATION => Err(CudaError::StreamCaptureIsolated),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_IMPLICIT => Err(CudaError::StreamCaptureImplicit),
            cudaError_enum::CUDA_ERROR_CAPTURED_EVENT => Err(CudaError::CapturedEvent),
            cudaError_enum::CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD => Err(CudaError::StreamCaptureWrongThread),
            cudaError_enum::CUDA_ERROR_TIMEOUT => Err(CudaError::Timeout),
            cudaError_enum::CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE => Err(CudaError::GraphExecUpdateFailure),
            cudaError_enum::CUDA_ERROR_EXTERNAL_DEVICE => Err(CudaError::ExternalDevice),
            cudaError_enum::CUDA_ERROR_INVALID_CLUSTER_SIZE => Err(CudaError::InvalidClusterSize),
            cudaError_enum::CUDA_ERROR_FUNCTION_NOT_LOADED => Err(CudaError::FunctionNotLoaded),
            cudaError_enum::CUDA_ERROR_INVALID_RESOURCE_TYPE => Err(CudaError::InvalidResourceType),
            cudaError_enum::CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION => Err(CudaError::InvalidResourceConfiguration),
            cudaError_enum::CUDA_ERROR_UNKNOWN => Err(CudaError::UnknownError),
        }
    }
}
