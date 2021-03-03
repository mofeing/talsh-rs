use std::os::raw::*;

const MAX_TENSOR_RANK: usize = 32;
const MAX_TENSOR_OPERANDS: usize = 4;
const MAX_CONTRACTION_PATTERN_LEN: usize = 1024;
const MAX_MLNDS_PER_TENS: usize = 4;

/// Tensor signature.
#[repr(C)]
pub struct talsh_tens_signature_t {
    /// Tensor rank (number of dimensions).
    ///
    /// >= 0; -1, empty
    pub num_dim: c_int,

    /// Tensor signature.
    ///
    /// An array of size <num_dim> (long integers).
    pub offsets: *mut usize,
}

/// Tensor shape.
#[repr(C)]
pub struct talsh_tens_shape_t {
    /// Rank
    pub num_dim: c_int,
    /// Extents
    pub dims: *mut c_int,
    /// Dividers
    pub divs: *mut c_int,
    /// Groups
    pub grps: *mut c_int,
}

/// Tensor data descriptor
#[repr(C)]
pub struct talsh_tens_data_t {
    pub base: *mut c_void,
    pub volume: usize,
    pub data_kind: c_int,
}

/// Dense tensor block
#[repr(C)]
pub struct talsh_tens_dense_t {
    pub num_dim: c_int,
    pub data_kind: c_int,
    pub body: *mut c_void,
    pub bases: [usize; MAX_TENSOR_RANK],
    pub dims: [usize; MAX_TENSOR_RANK],
}

/// Device resource (occupied by a tensor block)
#[repr(C)]
pub struct talsh_dev_rsc_t {
    pub dev_id: c_int,
    pub gmem_p: *mut c_void,
    pub buf_entry: c_int,
    pub mem_attached: c_int,
}

#[link(name = "talsh")]
extern "C" {
    pub fn tens_valid_data_kind(datk: c_int, datk_size: *mut c_int) -> c_int;
    pub fn tens_valid_data_kind_(datk: c_int, datk_size: *mut c_int) -> c_int;
    pub fn permutation_trivial(perm_len: c_int, perm: *mut c_int, base: c_int) -> c_int;
    pub fn get_contr_pattern_sym(
        rank_left: *mut c_int,
        rank_right: *mut c_int,
        conj_bits: *mut c_int,
        cptrn_dig: *mut c_int,
        cptrn_sym: *mut c_char,
        cpl: *mut c_int,
        ierr: *mut c_int,
    );
    pub fn get_contr_permutations(
        gemm_tl: c_int,
        gemm_tr: c_int,
        lrank: c_int,
        rrank: c_int,
        cptrn: *mut c_int,
        conj_bits: c_int,
        dprm: *mut c_int,
        lprm: *mut c_int,
        rprm: *mut c_int,
        ncd: *mut c_int,
        nlu: *mut c_int,
        nru: *mut c_int,
        ierr: *mut c_int,
    );
    // #ifdef USE_CUTENSOR
    // int get_contr_pattern_cutensor(const int * dig_ptrn, int drank, int32_t * ptrn_d, int lrank, int32_t * ptrn_l, int rrank, int32_t * ptrn_r);
    // #endif
    pub fn tens_elem_offset_f(num_dim: c_uint, dims: *mut c_uint, mlndx: *mut c_uint) -> usize;
    pub fn tens_elem_mlndx_f(offset: usize, num_dim: c_uint, dims: *mut c_uint, mlndx: *mut c_uint);
    pub fn argument_coherence_get_value(
        coh_ctrl: c_uint,
        tot_args: c_uint,
        arg_num: c_uint,
    ) -> c_uint;
    pub fn argument_coherence_set_value(
        coh_ctrl: *mut c_uint,
        tot_args: c_uint,
        arg_num: c_uint,
        coh_val: c_uint,
    ) -> c_int;

    // Device id conversion:
    pub fn valid_device_kind(dev_kind: c_int) -> c_int;
    pub fn encode_device_id(dev_kind: c_int, dev_num: c_int) -> c_int;
    pub fn decode_device_id(dev_id: c_int, dev_kind: *mut c_int) -> c_int;

    // Device resource management:
    pub fn tensDevRsc_create(drsc: *mut *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_clean(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_is_empty(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_same(drsc0: *const talsh_dev_rsc_t, drsc1: *const talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_clone(
        drsc_in: *const talsh_dev_rsc_t,
        drsc_out: *mut talsh_dev_rsc_t,
    ) -> c_int;
    pub fn tensDevRsc_attach_mem(
        drsc: *mut talsh_dev_rsc_t,
        dev_id: c_int,
        mem_p: *mut c_void,
        buf_entry: c_int,
    ) -> c_int;
    pub fn tensDevRsc_detach_mem(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_allocate_mem(
        drsc: *mut talsh_dev_rsc_t,
        dev_id: c_int,
        mem_size: usize,
        in_arg_buf: c_int,
    ) -> c_int;
    pub fn tensDevRsc_free_mem(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_get_gmem_ptr(drsc: *mut talsh_dev_rsc_t, gmem_p: *mut *mut c_void) -> c_int;
    pub fn tensDevRsc_device_id(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_release_all(drsc: *mut talsh_dev_rsc_t) -> c_int;
    pub fn tensDevRsc_destroy(drsc: *mut talsh_dev_rsc_t) -> c_int;

    /// C tensor block API:
    ///  Tensor signature:
    pub fn tensSignature_create(tsigna: *mut *mut talsh_tens_signature_t) -> c_int;
    pub fn tensSignature_clean(tsigna: *mut talsh_tens_signature_t) -> c_int;
    pub fn tensSignature_construct(
        tsigna: *mut talsh_tens_signature_t,
        rank: c_int,
        offsets: *const usize,
    ) -> c_int;
    pub fn tensSignature_destruct(tsigna: *mut talsh_tens_signature_t) -> c_int;
    pub fn tensSignature_destroy(tsigna: *mut talsh_tens_signature_t) -> c_int;

    ///  Tensor shape:
    pub fn tensShape_create(tshape: *mut *mut talsh_tens_shape_t) -> c_int;
    pub fn tensShape_clean(tshape: *mut talsh_tens_shape_t) -> c_int;
    pub fn tensShape_construct(
        tshape: *mut talsh_tens_shape_t,
        pinned: c_int,
        rank: c_int,
        dims: *const c_int,
        divs: *const c_int,
        grps: *const c_int,
    ) -> c_int;
    pub fn tensShape_destruct(tshape: *mut talsh_tens_shape_t) -> c_int;
    pub fn tensShape_destroy(tshape: *mut talsh_tens_shape_t) -> c_int;
    pub fn tensShape_volume(tshape: *const talsh_tens_shape_t) -> usize;
    pub fn tensShape_rank(tshape: *const talsh_tens_shape_t) -> c_int;
    pub fn tensShape_reshape(
        tshape: *mut talsh_tens_shape_t,
        rank: c_int,
        dims: *const c_int,
        divs: *const c_int,
        grps: *const c_int,
    ) -> c_int;
    pub fn tensShape_print(tshape: *const talsh_tens_shape_t);
}

#[repr(C)]
pub struct talsh_tens_t {
    pub shape_p: *mut talsh_tens_shape_t,
    pub dev_rsc: *mut talsh_dev_rsc_t,
    pub data_kind: *mut c_int,
    pub avail: *mut c_int,
    pub dev_rsc_len: c_int,
    pub ndev: c_int,
}

#[repr(C)]
pub struct talsh_tens_slice_t {
    pub tensor: *mut talsh_tens_t,
    pub bases: talsh_tens_signature_t,
    pub shape: talsh_tens_shape_t,
}

#[repr(C)]
pub struct talshTensArg_t {
    pub tens_p: *mut talsh_tens_t,
    pub source_image: c_int,
}

#[repr(C)]
pub struct talsh_task_t {
    task_p: *mut c_void,
    task_error: c_int,
    dev_kind: c_int,
    data_kind: c_int,
    coherence: c_int,
    num_args: c_int,
    tens_args: [talshTensArg_t; MAX_TENSOR_OPERANDS],
    data_vol: c_double,
    flops: c_double,
    exec_time: c_double,
}

#[repr(C)]
pub struct talsh_tens_op_t {
    opkind: c_int,
    data_kind: c_int,
    num_args: c_uint,
    tens_slice: [talsh_tens_slice_t; MAX_TENSOR_OPERANDS],
    symb_pattern: *const c_char,
    alpha_real: c_double,
    alpha_imag: c_double,
    tens_arg: [talsh_tens_t; MAX_TENSOR_OPERANDS],
    task_handle: talsh_task_t,
    exec_dev_id: c_int,
    stage: c_int,
    time_started: c_double,
    time_scheduled: c_double,
    time_completed: c_double,
    time_finished: c_double,
}

#[link(name = "talsh")]
extern "C" {
    pub fn talshValidDataKind(datk: c_int, datk_size: *mut c_int) -> c_int;

    // TAL-SH control API
    /// Initialize TAL-SH
    pub fn talshInit(
        host_buf_size: *mut usize,
        host_arg_max: *mut c_int,
        ngpus: c_int,
        gpu_list: *mut c_int,
        nmics: c_int,
        mic_list: *mut c_int,
        namds: c_int,
        amd_list: *mut c_int,
    ) -> c_int;

    /// Shutdown TAL-SH
    pub fn talshShudown() -> c_int;

    /// Set the memory allocation policy on Host
    pub fn talshSetMemAllocPolicyHost(mem_policy: c_int, fallback: c_int, ierr: c_int);

    /// Enable fast math on a given device
    pub fn talshEnableFastMath(dev_kind: c_int, dev_id: c_int) -> c_int;

    // Disable fast math on a given device:
    pub fn talshDisableFastMath(dev_kind: c_int, dev_id: c_int) -> c_int;

    // Query fast math on a given device:
    pub fn talshQueryFastMath(dev_kind: c_int, dev_id: c_int) -> c_int;

    /// Get on-node device count:
    pub fn talshDeviceCount(dev_kind: c_int, dev_count: *mut c_int) -> c_int;

    /// Get the flat device Id
    pub fn talshFlatDevId(dev_kind: c_int, dev_num: c_int) -> c_int;

    /// Get the kind-specific device Id:
    pub fn talshKindDevId(dev_id: c_int, dev_kind: *mut c_int) -> c_int;

    /// Query the state of a device:
    pub fn talshDeviceState(dev_num: c_int, dev_kind: c_int) -> c_int;
    pub fn talshDeviceState_(dev_num: c_int, dev_kind: c_int) -> c_int;

    /// Find the least busy device:
    pub fn talshDeviceBusyLeast(dev_kind: c_int) -> c_int;
    pub fn talshDeviceBusyLeast_(dev_kind: c_int) -> c_int;

    /// Determine the optimal execution device for given tensor operands:
    pub fn talshDetermineOptimalDevice(
        tens0: *const talsh_tens_t,
        tens1: *const talsh_tens_t,
        tens2: *const talsh_tens_t,
    ) -> c_int;

    /// Query device memory size (bytes):
    pub fn talshDeviceMemorySize(dev_num: c_int, dev_kind: c_int) -> usize;
    pub fn talshDeviceMemorySize_(dev_num: c_int, dev_kind: c_int) -> usize;

    /// Query device argument buffer size (bytes):
    pub fn talshDeviceBufferSize(dev_num: c_int, dev_kind: c_int) -> usize;
    pub fn talshDeviceBufferSize_(dev_num: c_int, dev_kind: c_int) -> usize;

    /// Query device max tensor size (bytes):
    pub fn talshDeviceTensorSize(dev_num: c_int, dev_kind: c_int) -> usize;
    pub fn talshDeviceTensorSize_(dev_num: c_int, dev_kind: c_int) -> usize;

    /// Query the amount of free memory in an argument buffer on a given device (bytes):
    pub fn talshDeviceBufferFreeSize(dev_num: c_int, dev_kind: c_int) -> usize;
    pub fn talshDeviceBufferFreeSize_(dev_num: c_int, dev_kind: c_int) -> usize;

    /// Query the current executed flop count:
    pub fn talshDeviceGetFlops(dev_kind: c_int, dev_id: c_int) -> c_double;

    /// Start memory manager log:
    pub fn talshMemManagerLogStart();

    /// Finish memory manager log:
    pub fn talshMemManagerLogFinish();

    /// Start basic tensor operation logging:
    pub fn talshTensorOpLogStart();

    /// Finish basic tensor operation logging:
    pub fn talshTensorOpLogFinish();

    /// Print TAL-SH statistics for specific devices:
    pub fn talshStats(dev_id: c_int, dev_kind: c_int) -> c_int;
    pub fn talshStats_(dev_id: c_int, dev_kind: c_int) -> c_int;

    // TAL-SH tensor block API:
    /// Create an empty tensor block:
    pub fn talshTensorCreate(tens_block: *mut *mut talsh_tens_t) -> c_int;

    /// Clean an undefined tensor block (default constructor):
    pub fn talshTensorClean(tens_block: *mut talsh_tens_t) -> c_int;

    /// Check whether a tensor block is empty (clean):
    pub fn talshTensorIsEmpty(tens_block: *const talsh_tens_t) -> c_int;

    /// Construct a tensor block:
    // fn talshTensorConstruct(tens_block: *mut talsh_tens_t, data_kind: c_int, tens_rank: c_int, tens_dims: *const c_int, dev_id: c_int, ext_mem: *mut c_void, in_hab: c_int, init_method: talsh_tens_init_i , init_val_real: c_double, init_val_imag: c_double) -> c_int;
    // fn talshTensorConstruct_(tens_block: *mut talsh_tens_t, data_kind: c_int, tens_rank: c_int, tens_dims: *const c_int, dev_id: c_int,
    ///    ext_mem: *mut c_void, in_hab: c_int, init_method: talsh_tens_init_i , init_val_real: c_double, init_val_imag: c_double) -> c_int;

    /// Import external data for the tensor body:
    pub fn talshTensorImportData(
        tens_block: *mut talsh_tens_t,
        data_kind: c_int,
        ext_data: *const c_void,
    ) -> c_int;

    /// Destruct a tensor block:
    pub fn talshTensorDestruct(tens_block: *mut talsh_tens_t) -> c_int;

    /// Destroy a tensor block:
    pub fn talshTensorDestroy(tens_block: *mut talsh_tens_t) -> c_int;

    /// Get the tensor block rank (number of dimensions):
    pub fn talshTensorRank(tens_block: *const talsh_tens_t) -> c_int;

    /// Get the tensor block volume (number of elements per image):
    pub fn talshTensorVolume(tens_block: *const talsh_tens_t) -> usize;

    /// Get the size of all tensor images in bytes:
    pub fn talshTensorSizeAllImages(
        tens_block: *const talsh_tens_t,
        num_images: *mut c_int,
    ) -> usize;

    /// Get tensor dimension extents:
    pub fn talshTensorDimExtents(tens_block: *const talsh_tens_t, rank: *mut c_int)
        -> *const c_int;

    /// Get the shape of the tensor block:
    pub fn talshTensorShape(
        tens_block: *const talsh_tens_t,
        tens_shape: *mut talsh_tens_shape_t,
    ) -> c_int;

    /// Get the data kind of each tensor image:
    pub fn talshTensorDataKind(
        tens_block: *const talsh_tens_t,
        num_images: *mut c_int,
        data_kinds: *mut c_int,
    ) -> c_int;

    /// Reshape a tensor to a compatible shape (same volume):
    pub fn talshTensorReshape(
        tens_block: *mut talsh_tens_t,
        tens_rank: c_int,
        tens_dims: *const c_int,
    ) -> c_int;

    /// Query whether the tensor block is currently in use:
    pub fn talshTensorInUse(tens_block: *const talsh_tens_t) -> c_int;

    /// Query the presence of the tensor block on device(s):
    pub fn talshTensorPresence(
        tens_block: *const talsh_tens_t,
        ncopies: *mut c_int,
        copies: *mut c_int,
        data_kinds: *mut c_int,
        dev_kind: c_int,
        dev_id: c_int,
    ) -> c_int;
    pub fn talshTensorPresence_(
        tens_block: *const talsh_tens_t,
        ncopies: *mut c_int,
        copies: *mut c_int,
        data_kinds: *mut c_int,
        dev_kind: c_int,
        dev_id: c_int,
    ) -> c_int;

    /// Get access to the tensor body image for a subsequent initialization:
    pub fn talshTensorGetBodyAccess(
        tens_block: *mut talsh_tens_t,
        body_p: *mut *mut c_void,
        data_kind: c_int,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;
    pub fn talshTensorGetBodyAccess_(
        tens_block: *mut talsh_tens_t,
        body_p: *mut *mut c_void,
        data_kind: c_int,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;

    /// Get access to the tensor body image read-only:
    pub fn talshTensorGetBodyAccessConst(
        tens_block: *const talsh_tens_t,
        body_p: *mut *const c_void,
        data_kind: c_int,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;

    /// Get the scalar value of the rank-0 tensor:
    pub fn talshTensorGetScalar(
        tens_block: *mut talsh_tens_t,
        scalar_real: *mut c_double,
        scalar_imag: *mut c_double,
    ) -> c_int;

    /// Print the shape of a tensor block:
    pub fn talshTensorPrint(tens_block: *const talsh_tens_t);

    /// Print the information on a tensor block:
    pub fn talshTensorPrintInfo(tens_block: *const talsh_tens_t);

    /// Print tensor elements larger by absolute value than some threshold:
    pub fn talshTensorPrintBody(tens_block: *const talsh_tens_t, thresh: c_double);

    // TAL-SH tensor slice API:
    /// Create an empty TAL-SH tensor slice:
    pub fn talshTensorSliceCreate(slice: *mut *mut talsh_tens_slice_t) -> c_int;

    /// Clean an undefined TAL-SH tensor slice:
    pub fn talshTensorSliceClean(slice: *mut talsh_tens_slice_t) -> c_int;

    /// Construct a TAL-SH tensor slice:
    pub fn talshTensorSliceConstruct(
        slice: *mut talsh_tens_slice_t,
        tensor: *const talsh_tens_t,
        offsets: *const usize,
        dims: *const c_int,
        divs: *const c_int,
        grps: *const c_int,
    ) -> c_int;

    /// Get the volume of the TAL-SH tensor slice:
    pub fn talshTensorSliceVolume(slice: *const talsh_tens_slice_t) -> usize;

    /// Destruct a TAL-SH tensor slice:
    pub fn talshTensorSliceDestruct(slice: *mut talsh_tens_slice_t) -> c_int;

    /// Destroy a TAL-SH tensor slice:
    pub fn talshTensorSliceDestroy(slice: *mut talsh_tens_slice_t) -> c_int;

    // TAL-SH task API:
    /// Create a clean (defined-empty) TAL-SH task:
    pub fn talshTaskCreate(talsh_task: *mut *mut talsh_task_t) -> c_int;

    /// Clean an undefined TAL-SH task:
    pub fn talshTaskClean(talsh_task: *mut talsh_task_t) -> c_int;

    /// Query whether a TAL-SH task is empty:
    pub fn talshTaskIsEmpty(talsh_task: *const talsh_task_t) -> c_int;

    /// Destruct a TAL-SH task:
    pub fn talshTaskDestruct(talsh_task: *mut talsh_task_t) -> c_int;

    /// Destroy a TAL-SH task:
    pub fn talshTaskDestroy(talsh_task: *mut talsh_task_t) -> c_int;

    /// Get the id of the device the TAL-SH task is scheduled on:
    pub fn talshTaskDevId(talsh_task: *mut talsh_task_t, dev_kind: *mut c_int) -> c_int;
    pub fn talshTaskDevId_(talsh_task: *mut talsh_task_t, dev_kind: *mut c_int) -> c_int;

    /// Get the argument coherence control value used in the TAL-SH task:
    pub fn talshTaskArgCoherence(talsh_task: *const talsh_task_t) -> c_int;

    /// Get the tensor arguments used in the TAL-SH task:
    pub fn talshTaskTensArgs(
        talsh_task: *const talsh_task_t,
        num_args: *mut c_int,
    ) -> *const talshTensArg_t;

    /// Get the TAL-SH task status:
    pub fn talshTaskStatus(talsh_task: *mut talsh_task_t) -> c_int;

    /// Check whether a TAL-SH task has completed:
    pub fn talshTaskComplete(
        talsh_task: *mut talsh_task_t,
        stats: *mut c_int,
        ierr: *mut c_int,
    ) -> c_int;

    /// Wait upon a completion of a TAL-SH task:
    pub fn talshTaskWait(talsh_task: *mut talsh_task_t, stats: *mut c_int) -> c_int;

    /// Wait upon a completion of multiple TAL-SH tasks:
    pub fn talshTasksWait(
        ntasks: c_int,
        talsh_tasks: *mut talsh_task_t,
        stats: *mut c_int,
    ) -> c_int;

    /// Get the TAL-SH task timings:
    pub fn talshTaskTime(
        talsh_task: *mut talsh_task_t,
        total: *mut c_double,
        comput: *mut c_double,
        input: *mut c_double,
        output: *mut c_double,
        mmul: *mut c_double,
    ) -> c_int;
    pub fn talshTaskTime_(
        talsh_task: *mut talsh_task_t,
        total: *mut c_double,
        comput: *mut c_double,
        input: *mut c_double,
        output: *mut c_double,
        mmul: *mut c_double,
    ) -> c_int;

    /// Print TAL-SH task info:
    pub fn talshTaskPrint(talsh_task: *const talsh_task_t);

    // TAL-SH tensor operations API:
    /// Create an empty tensor operation:
    pub fn talshTensorOpCreate(tens_op: *mut *mut talsh_tens_op_t) -> c_int;

    /// Clean an undefined tensor operation:
    pub fn talshTensorOpClean(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Set a tensor operation argument (tensor slice):
    pub fn talshTensorOpSetArgument(
        tens_op: *mut talsh_tens_op_t,
        tensor: *const talsh_tens_t,
        offsets: *const usize,
        dims: *const c_int,
    ) -> c_int;

    /// Specify the kind of the tensor operation:
    pub fn talshTensorOpSpecify(
        tens_op: *mut talsh_tens_op_t,
        operation_kind: c_int,
        data_kind: c_int,
        symbolic_pattern: *const c_char,
        prefactor_real: c_double,
        prefactor_imag: c_double,
    ) -> c_int;

    /// Preset execution device:
    pub fn talshTensorOpSetExecDevice(
        tens_op: *mut talsh_tens_op_t,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;

    /// Activate tensor operation for subsequent processing (resources acquired):
    pub fn talshTensorOpActivate(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Load input (extract input tensor slices):
    pub fn talshTensorOpLoadInput(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Schedule tensor operation for execution of a given device:
    pub fn talshTensorOpExecute(
        tens_op: *mut talsh_tens_op_t,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;

    /// Test for tensor operation completion:
    pub fn talshTensorOpTest(
        tens_op: *mut talsh_tens_op_t,
        completed: *mut c_int,
        wait: c_int,
    ) -> c_int;

    /// Store output (insert/accumulate output tensor slice):
    pub fn talshTensorOpStoreOutput(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Deactivate tensor operation (resources released):
    pub fn talshTensorOpDeactivate(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Destruct tensor operation (back to an empty state):
    pub fn talshTensorOpDestruct(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Destroy tensor operation:
    pub fn talshTensorOpDestroy(tens_op: *mut talsh_tens_op_t) -> c_int;

    /// Progress tensor operation execution:
    pub fn talshTensorOpProgress(tens_op: *mut talsh_tens_op_t, done: *mut c_int) -> c_int;

    /// Get tensor argument volume:
    pub fn talshTensorOpGetArgVolume(tens_op: *const talsh_tens_op_t, arg_num: c_uint) -> usize;

    /// Get tensor argument size in bytes:
    pub fn talshTensorOpGetArgSize(tens_op: *const talsh_tens_op_t, arg_num: c_uint) -> usize;

    /// Tensor operation byte count (memory requirements):
    pub fn talshTensorOpGetByteCount(
        tens_op: *const talsh_tens_op_t,
        element_size: c_uint,
    ) -> c_double;

    /// Tensor operation floating point count (compute requirements):
    pub fn talshTensorOpGetFlopCount(tens_op: *const talsh_tens_op_t) -> c_double;

    /// Tensor operation arithmetic intensity:
    pub fn talshTensorOpGetIntensity(tens_op: *const talsh_tens_op_t) -> c_double;

    /// Tensor operation decomposition into two sub-operations:
    //in: parent tensor operation (defined on entrance)
    //inout: children tensor operation 1 (empty on entrance)
    //inout: children tensor operation 2 (empty on entrance)
    pub fn talshTensorOpDecompose2(
        tens_op: *const talsh_tens_op_t,
        child_op1: *mut talsh_tens_op_t,
        child_op2: *mut talsh_tens_op_t,
    ) -> c_int;

    /// Print tensor operation:
    pub fn talshTensorOpPrint(tens_op: *const talsh_tens_op_t);

    /// Place a tensor block on a specific device:
    pub fn talshTensorPlace(
        tens: *mut talsh_tens_t, //inout: tensor block
        dev_id: c_int,           //in: device id (flat or kind-specific)
        dev_kind: c_int,         //in: device kind (if present, <dev_id> is kind-specific)
        dev_mem: *mut c_void,    //in: externally provided target device memory pointer
        copy_ctrl: c_int,        //in: copy control (COPY_X), defaults to COPY_M
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorPlace_(
        tens: *mut talsh_tens_t,
        dev_id: c_int,
        dev_kind: c_int,
        dev_mem: *mut c_void,
        copy_ctrl: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Discard a tensor block on a specific device:
    pub fn talshTensorDiscard(
        tens: *mut talsh_tens_t, //inout: tensor block
        dev_id: c_int,           //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    pub fn talshTensorDiscard_(tens: *mut talsh_tens_t, dev_id: c_int, dev_kind: c_int) -> c_int;
    /// Discard a tensor block on all devices except a specific device:
    pub fn talshTensorDiscardOther(
        tens: *mut talsh_tens_t, //inout: tensor block
        dev_id: c_int,           //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    pub fn talshTensorDiscardOther_(
        tens: *mut talsh_tens_t,
        dev_id: c_int,
        dev_kind: c_int,
    ) -> c_int;
    /// Tensor initialization:
    pub fn talshTensorInit(
        dtens: *mut talsh_tens_t, //inout: tensor block
        val_real: c_double,       //in: initialization value (real part)
        val_imag: c_double,       //in: initialization value (imaginary part)
        dev_id: c_int,            //in: device id (flat or kind-specific)
        dev_kind: c_int,          //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,         //in: copy control (COPY_X), defaults to COPY_M
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorInit_(
        dtens: *mut talsh_tens_t,
        val_real: c_double,
        val_imag: c_double,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor slicing:
    pub fn talshTensorSlice(
        dtens: *mut talsh_tens_t, //inout: destination tensor block (tensor slice)
        ltens: *mut talsh_tens_t, //inout: source tensor block
        offsets: *const c_int,    //in: base offsets of the slice (0-based numeration)
        dev_id: c_int,            //in: device id (flat or kind-specific)
        dev_kind: c_int,          //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,         //in: copy control (COPY_XX), defaults to COPY_MT
        accumulative: c_int,      //in: accumulate in VS overwrite destination tensor: [YEP|NOPE]
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorSlice_(
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        offsets: *const c_int,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        accumulative: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor insertion:
    pub fn talshTensorInsert(
        dtens: *mut talsh_tens_t, //inout: destination tensor block
        ltens: *mut talsh_tens_t, //inout: source tensor block (tensor slice)
        offsets: *const c_int,    //in: base offsets of the slice (0-based numeration)
        dev_id: c_int,            //in: device id (flat or kind-specific)
        dev_kind: c_int,          //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,         //in: copy control (COPY_XX), defaults to COPY_MT
        accumulative: c_int,      //in: accumulate in VS overwrite destination tensor: [YEP|NOPE]
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorInsert_(
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        offsets: *const c_int,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        accumulative: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor copy (with an optional permutation of indices):
    pub fn talshTensorCopy(
        cptrn: *const c_char, //in: C-string: symbolic copy pattern, e.g. "D(a,b,c,d)=L(c,d,b,a)"
        dtens: *mut talsh_tens_t, //inout: destination tensor block
        ltens: *mut talsh_tens_t, //inout: source tensor block
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,      //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,     //in: copy control (COPY_XX), defaults to COPY_MT
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorCopy_(
        cptrn: *const c_char,
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor addition:
    pub fn talshTensorAdd(
        cptrn: *const c_char, //in: C-string: symbolic addition pattern, e.g. "D(a,b,c,d)+=L(c,d,b,a)"
        dtens: *mut talsh_tens_t, //inout: destination tensor block
        ltens: *mut talsh_tens_t, //inout: source tensor block
        scale_real: c_double, //in: scaling value (real part), defaults to 1
        scale_imag: c_double, //in: scaling value (imaginary part), defaults to 0
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,      //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,     //in: copy control (COPY_XX), defaults to COPY_MT
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task handle
    pub fn talshTensorAdd_(
        cptrn: *const c_char,
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        scale_real: c_double,
        scale_imag: c_double,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor contraction:
    pub fn talshTensorContract(
        cptrn: *const c_char, //in: C-string: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //inout: destination tensor block
        ltens: *mut talsh_tens_t, //inout: left source tensor block
        rtens: *mut talsh_tens_t, //inout: right source tensor block
        scale_real: c_double, //in: scaling value (real part), defaults to 1
        scale_imag: c_double, //in: scaling value (imaginary part), defaults to 0
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,      //in: device kind (if present, <dev_id> is kind-specific)
        copy_ctrl: c_int,     //in: copy control (COPY_XXX), defaults to COPY_MTT
        accumulative: c_int, //in: accumulate in (default) VS overwrite destination tensor: [YEP|NOPE]
        talsh_task: *mut talsh_task_t,
    ) -> c_int; //inout: TAL-SH task (must be clean)
    pub fn talshTensorContract_(
        cptrn: *const c_char,
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        rtens: *mut talsh_tens_t,
        scale_real: c_double,
        scale_imag: c_double,
        dev_id: c_int,
        dev_kind: c_int,
        copy_ctrl: c_int,
        accumulative: c_int,
        talsh_task: *mut talsh_task_t,
    ) -> c_int;
    /// Tensor contraction (extra large):
    pub fn talshTensorContractXL(
        cptrn: *const c_char, //in: C-string: symbolic contraction pattern, e.g. "D(a,b,c,d)+=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //inout: destination tensor block
        ltens: *mut talsh_tens_t, //inout: left source tensor block
        rtens: *mut talsh_tens_t, //inout: right source tensor block
        scale_real: c_double, //in: scaling value (real part), defaults to 1
        scale_imag: c_double, //in: scaling value (imaginary part), defaults to 0
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,      //in: device kind (if present, <dev_id> is kind-specific)
        accumulative: c_int,
    ) -> c_int; //in: accumulate in (default) VS overwrite destination tensor: [YEP|NOPE]
    pub fn talshTensorContractXL_(
        cptrn: *const c_char,
        dtens: *mut talsh_tens_t,
        ltens: *mut talsh_tens_t,
        rtens: *mut talsh_tens_t,
        scale_real: c_double,
        scale_imag: c_double,
        dev_id: c_int,
        dev_kind: c_int,
        accumulative: c_int,
    ) -> c_int;
    /// Tensor decomposition via SVD:
    ///  Meaning of parameter <absorb>:
    ///   'N': No absorption of stens;
    ///   'L': stens will be absorbed into ltens;
    ///   'R': stens will be absorbed into rtens;
    ///   'S': square root of stens will be absorbed into both ltens and rtens;
    pub fn talshTensorDecomposeSVD(
        cptrn: *const c_char, //in: C-string: symbolic decomposition pattern, e.g. "D(a,b,c,d)=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //in: tensor block to be decomposed
        ltens: *mut talsh_tens_t, //inout: left tensor factor
        rtens: *mut talsh_tens_t, //inout: right tensor factor
        stens: *mut talsh_tens_t, //inout: middle tensor factor (singular values), may be empty on entrance
        absorb: c_char, //in: whether or not to absorb the middle factor stens into other factors
        dev_id: c_int,  //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    /// Tensor decomposition via SVD with singular values absorbed into the left factor:
    pub fn talshTensorDecomposeSVDL(
        cptrn: *const c_char, //in: C-string: symbolic decomposition pattern, e.g. "D(a,b,c,d)=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //in: tensor block to be decomposed
        ltens: *mut talsh_tens_t, //inout: left tensor factor with absorbed singular values
        rtens: *mut talsh_tens_t, //inout: right tensor factor
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    /// Tensor decomposition via SVD with singular values absorbed into the right factor:
    pub fn talshTensorDecomposeSVDR(
        cptrn: *const c_char, //in: C-string: symbolic decomposition pattern, e.g. "D(a,b,c,d)=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //in: tensor block to be decomposed
        ltens: *mut talsh_tens_t, //inout: left tensor factor
        rtens: *mut talsh_tens_t, //inout: right tensor factor with absorbed singular values
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    /// Tensor decomposition via SVD with symmetrically absorbed square-root singular values into both factors:
    pub fn talshTensorDecomposeSVDLR(
        cptrn: *const c_char, //in: C-string: symbolic decomposition pattern, e.g. "D(a,b,c,d)=L(c,i,j,a)*R(b,j,d,i)"
        dtens: *mut talsh_tens_t, //in: tensor block to be decomposed
        ltens: *mut talsh_tens_t, //inout: left tensor factor with absorbed square-root singular values
        rtens: *mut talsh_tens_t, //inout: right tensor factor with absorbed square-root singular values
        dev_id: c_int,            //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    /// Tensor orthogonalization via SVD (D=LR+ with singular values reset to unity):
    ///  The symbolic tensor decomposition (contraction) pattern must only have one contracted index,
    ///  its dimension being equal to the minimum of the left and right uncontracted dimension volumes:
    pub fn talshTensorOrthogonalizeSVD(
        cptrn: *const c_char, //in: C-string: symbolic decomposition pattern, e.g. "D(a,b,c,d)=L(c,i,a)*R(b,d,i)"
        dtens: *mut talsh_tens_t, //inout: on entrance tensor block to be orthogonalized, on exit orthogonalized tensor block
        dev_id: c_int,            //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
    /// Tensor orthogonalization via MGS (D=Q from QR):
    pub fn talshTensorOrthogonalizeMGS(
        dtens: *mut talsh_tens_t, //inout: on entrance tensor block to be orthogonalized, on exit orthogonalized tensor block
        num_iso_dims: c_int,      //in: number of the isometric tensor dimensions
        iso_dims: *mut c_int, //in: ordered list of the isometric tensor dimensions (tensor dimension numeration starts from 0)
        dev_id: c_int,        //in: device id (flat or kind-specific)
        dev_kind: c_int,
    ) -> c_int; //in: device kind (if present, <dev_id> is kind-specific)
                // TAL-SH debugging:
    /// 1-norm of the tensor body image on Host:
    pub fn talshTensorImageNorm1_cpu(talsh_tens: *const talsh_tens_t) -> c_double;

}
