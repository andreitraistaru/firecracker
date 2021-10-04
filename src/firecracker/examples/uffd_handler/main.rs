mod uffd_handler;

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::{Arc, Barrier};

fn firecracker_api_call(stream: &mut UnixStream, method_and_uri: &str, body: &str) {
    let request = format!(
        "{} HTTP/1.1\r\n\
         Content-Length: {}\r\n\
         Content-Type: application/json\r\n\r\n\
         {}",
        method_and_uri,
        body.len(),
        body
    );
    println!("API request:\n{}", request);
    stream
        .write_all(request.as_bytes())
        .expect("cannot send API request");

    let mut response = vec![0u8; 1024];
    let bytes_read = stream
        .read(&mut response[..])
        .expect("cannot read API response");
    response.resize(bytes_read, 0);
    println!(
        "API response of {} bytes:\n{}",
        bytes_read,
        String::from_utf8(response).unwrap()
    );
}

fn main() {
    println!("Connecting to Firecracker API.");

    // TODO: make paths configurable thru cmdline params.
    let snapshot_path = "./foo.image";
    let mem_file_path = "/tmp/foo.mem";
    let path_to_api_socket = "/tmp/firecracker-sb0.sock";
    let path_to_uffd_socket = "/tmp/firecracker-sb0-uffd.sock";

    let mut socket = UnixStream::connect(path_to_api_socket).expect("cannot connect");

    let barrier = Arc::new(Barrier::new(2));
    let uffd_sock_path = path_to_uffd_socket.to_string();
    let mem_fpath = mem_file_path.to_string();
    let uds_barrier = barrier.clone();

    let handle =
        std::thread::spawn(move || uffd_handler::run(uffd_sock_path, mem_fpath, uds_barrier));
    // Wait for uffd thread to start listening on uffd UDS before sending API call to fc.
    barrier.wait();
    println!("Sending Load Snap API call.");

    let body = format!(
        "\
        {{\
            \"snapshot_path\":\"{}\",\
            \"mem_backend_type\":\"UffdOverUDS\",\
            \"mem_backend_path\":\"{}\"\
        }}",
        snapshot_path, path_to_uffd_socket
    );
    firecracker_api_call(&mut socket, "PUT /snapshot/load", &body);

    println!("Sending Resume-VM API call.");
    firecracker_api_call(&mut socket, "PATCH /vm", "{\"state\":\"Resumed\"}");

    handle.join().expect("uffd thread crashed");
}
