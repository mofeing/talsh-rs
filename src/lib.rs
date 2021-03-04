use num_derive::FromPrimitive;
use talsh_sys::*;

#[derive(FromPrimitive)]
pub enum ErrorCode {
    Success = 0,
    Failure = -666,
    NotAvailable = -888,
    NotImplemented = -999,
    NotInitialized = 1000000,
    AlreadyInitialized = 1000001,
    InvalidArgs = 1000002,
    IntegerOverflow = 1000003,
    ObjectNotEmpty = 1000004,
    ObjectIsEmpty = 1000005,
    InProgress = 1000006,
    NotAllowed = 1000007,
    LimitExceeded = 1000008,
    NotFound = 1000009,
    ObjectBroken = 1000010,
    InvalidRequest = 1000011,
}

#[derive(FromPrimitive)]
enum TaskStatus {
    Error = 1999999,
    Empty = 2000000,
    Scheduled = 2000001,
    Started = 2000002,
    InputReady = 2000003,
    OutputReady = 2000004,
    Completed = 2000005,
}

#[derive(FromPrimitive)]
enum OperationKinds {
    Noop,
    Init = 68,
    Norm1 = 69,
    Norm2 = 70,
    Min = 71,
    Max = 72,
    Fold = 73,
    Unfold = 74,
    Slice = 75,
    Insert = 76,
    Copy = 77,
    Permute = 78,
    Scale = 79,
    Add = 80,
    Trace = 81,
    Contract = 82,
    Hadamard = 83,
    Khatrirao = 84,
}

#[derive(FromPrimitive)]
enum OperationStage {
    Undefined = -1,
    Empty = 0,
    Partial = 1,
    Defined = 2,
    Resourced = 3,
    Loaded = 4,
    Scheduled = 5,
    Completed = 6,
    Stored = 7,
    Retired = 8,
}
