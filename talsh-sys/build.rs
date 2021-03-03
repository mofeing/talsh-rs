use std::env;
use std::process::Command;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("make")
        .args(&["-C", "external/ExaTENSOR/TALSH"])
        .status()
        .expect("Failed during TAL-SH build");

    Command::new("cp")
        .arg("external/ExaTENSOR/TALSH/libtalsh.a")
        .arg(&out_dir)
        .status()
        .unwrap();

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=talsh");
}
