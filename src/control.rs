use crate::ErrorCode;
use ffi::{talshDeviceCount, talshDisableFastMath, talshShudown};
use num_traits::FromPrimitive;
use talsh_sys as ffi;

pub fn init() -> Result<(), ErrorCode> {
    todo!()
}

pub fn shutdown() -> Result<(), ErrorCode> {
    let err = ErrorCode::from_i32(unsafe { ffi::talshShudown() });
    match err {
        Some(ErrorCode::Success) => Ok(()),
        _ => Err(err.unwrap()),
    }
}

pub fn alloc_policy() -> Result<(), ErrorCode> {
    todo!()
}

pub fn enable_fast_math(dev_kind: i32, dev_id: i32) -> Result<(), ErrorCode> {
    let err = ErrorCode::from_i32(unsafe { ffi::talshEnableFastMath(dev_kind, dev_id) });
    match err {
        Some(ErrorCode::Success) => Ok(()),
        _ => Err(err.unwrap()),
    }
}

pub fn disable_fast_math(dev_kind: i32, dev_id: i32) -> Result<(), ErrorCode> {
    let err = ErrorCode::from_i32(unsafe { ffi::talshDisableFastMath(dev_kind, dev_id) });
    match err {
        Some(ErrorCode::Success) => Ok(()),
        _ => Err(err.unwrap()),
    }
}

pub fn query_fast_math() -> Result<bool, ErrorCode> {
    todo!()
}

pub fn device_count(dev_kind: i32) -> Result<i32, ErrorCode> {
    let mut dev_count: i32 = 0;
    let err = ErrorCode::from_i32(unsafe { ffi::talshDeviceCount(dev_kind, &mut dev_count) });
    match err {
        Some(ErrorCode::Success) => Ok(dev_count),
        _ => Err(err.unwrap()),
    }
}

pub fn flat_dev_id(dev_kind: i32, dev_num: i32) -> Result<i32, ErrorCode> {
    todo!()
}

pub fn kind_dev_id(dev_id: i32) -> Result<i32, ErrorCode> {
    todo!()
}

// pub fn device_state()
