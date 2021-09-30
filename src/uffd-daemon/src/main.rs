use libc::c_void;
use nix::poll::{poll, PollFd, PollFlags};
use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use nix::unistd::{sysconf, SysconfVar};
use serde::Deserialize;
use std::fs::File;
use std::io::Write;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::ptr;
use std::sync::{Arc, Barrier};
use userfaultfd::Uffd;
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

// copy/pasted from firecracker vmm
mod firecracker_imports {
    use super::Deserialize;
    /// This describes the mapping between Firecracker base virtual address and offset in the
    /// buffer or file backend for a guest memory region. It is used to tell an external
    /// process/thread where to populate the guest memory data for this range.
    ///
    /// E.g. Guest memory contents for a region of `size` bytes can be found in the backend
    /// at `offset` bytes from the beginning, and should be copied/populated into `base_host_address`.
    #[derive(Clone, Debug, Default, Deserialize)]
    pub struct RegionBackendMapping {
        /// Base host virtual address where the guest memory contents for this region
        /// should be copied/populated.
        pub base_h_va: u64,
        /// Region size.
        pub size: usize,
        /// Offset in the backend file/buffer where the region contents are.
        pub offset: u64,
    }
}
use firecracker_imports::RegionBackendMapping;

fn uffd_handler(regions: Vec<RegionBackendMapping>, uffd: Uffd, mem_file_path: PathBuf) {
    let page_size = sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap() as usize;

    let memsize = regions.iter().map(|r| r.size).sum();
    let file = File::open(mem_file_path).expect("cannot open memfile");
    assert_eq!(memsize as u64, file.metadata().unwrap().len());

    // Create a page that will be copied into the faulting region
    let memfile_buffer = unsafe {
        mmap(
            ptr::null_mut(),
            memsize,
            ProtFlags::PROT_READ,
            MapFlags::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
        .expect("mmap")
    };

    // Loop, handling incoming events on the userfaultfd file descriptor
    loop {
        // See what poll() tells us about the userfaultfd
        let pollfd = PollFd::new(uffd.as_raw_fd(), PollFlags::POLLIN);
        let nready = poll(&mut [pollfd], -1).expect("poll");

        println!("\nfault_handler_thread():");
        let revents = pollfd.revents().unwrap();
        println!(
            "    poll() returns: nready = {}; POLLIN = {}; POLLERR = {}",
            nready,
            revents.contains(PollFlags::POLLIN),
            revents.contains(PollFlags::POLLERR),
        );

        // Read an event from the userfaultfd
        let event = uffd
            .read_event()
            .expect("read uffd_msg")
            .expect("uffd_msg ready");

        // We expect only one kind of event; verify that assumption
        if let userfaultfd::Event::Pagefault { addr, .. } = event {
            // Display info about the page-fault event
            println!("    UFFD_EVENT_PAGEFAULT event: {:?}", event);
            let dst = (addr as usize & !(page_size as usize - 1)) as *mut c_void;
            println!(
                "    looking for file offset corresponding to hVA dst: {:?}",
                dst
            );
            let mut found_mapping = None;
            for r in regions.iter() {
                let fault_page_addr = dst as u64;
                if r.base_h_va <= fault_page_addr && fault_page_addr < r.base_h_va + r.size as u64 {
                    found_mapping = Some(r.clone());
                }
            }
            if let Some(r) = &found_mapping {
                println!("    found it in: {:?}", r);
            } else {
                panic!("could not find addr within region mappings");
            }

            let r = found_mapping.unwrap();
            let src = memfile_buffer as u64 + r.offset;
            // Populate whole region from backing mem-file.
            let copy = unsafe {
                uffd.copy(src as *const _, r.base_h_va as *mut _, r.size, true)
                    .expect("uffd copy")
            };
            println!("        (uffdio_copy.copy returned {})", copy);
        } else {
            panic!("Unexpected event on userfaultfd");
        }
    }
}

fn firecracker_api_call(sock: &mut UnixStream, method_and_uri: &str, body: &str) {
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
    sock.write_all(request.as_bytes()).unwrap();

    let mut response = vec![0u8; 1024];
    let (bytes_read, file) = sock.recv_with_fd(&mut response[..]).unwrap();
    let passed_fd = file.is_some();
    println!(
        "API response with {} bytes and {} FD:\n{}",
        bytes_read,
        passed_fd,
        String::from_utf8(response).unwrap()
    );

    // if let Some(mut file) = file {
    //     let mut contents = String::new();
    //     file.read_to_string(&mut contents)
    //         .expect("could not read file");
    //     println!("Passed FD contents: {}", contents);
    // }
}

fn uffd_thread(uffd_sock_path: String, mem_file_path: String, barrier: Arc<Barrier>) {
    let listener = UnixListener::bind(&uffd_sock_path).expect("cannot bind");

    println!("Bound UDS at: {:?}", uffd_sock_path);
    // Signal main thread we're ready and listening on uffd UDS.
    barrier.wait();

    println!("listening on uffd sock");
    let (stream, _) = listener.accept().expect("cannot listen");
    println!("new connection on uffd sock");

    let mut message_buf = vec![0u8; 1024];
    let (bytes_read, file) = stream
        .recv_with_fd(&mut message_buf[..])
        .expect("cannot recv_with_fd");
    message_buf.resize(bytes_read, 0);
    let passed_fd = file.is_some();

    let body = String::from_utf8(message_buf).unwrap();
    println!(
        "API response with {} bytes and {} FD:\n{:?}",
        bytes_read, passed_fd, body
    );

    let mem_desc = serde_json::from_str::<Vec<RegionBackendMapping>>(&body).expect("deser failed");
    let uffd = unsafe { Uffd::from_raw_fd(file.unwrap().into_raw_fd()) };

    uffd_handler(mem_desc, uffd, mem_file_path.into());
    println!("Uffd thread done!");
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
    // std::thread::spawn(move || uffd_handler(uffd_backend_desc, mem_file_path));
    let uffd_sock_path = path_to_uffd_socket.to_string();
    let mem_fpath = mem_file_path.to_string();
    let uds_barrier = barrier.clone();

    let handle = std::thread::spawn(move || uffd_thread(uffd_sock_path, mem_fpath, uds_barrier));
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

    firecracker_api_call(&mut socket, "PATCH /vm", "{\"state\":\"Resumed\"}");

    handle.join().expect("uffd thread crashed");
}
