use libc::c_void;
use nix::poll::{poll, PollFd, PollFlags};
use nix::sys::mman::{mmap, MapFlags, ProtFlags};
use nix::unistd::{sysconf, SysconfVar};
use serde::Deserialize;
use std::fs::File;
use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::ptr;
use std::sync::{Arc, Barrier};
use userfaultfd::Uffd;
use utils::sock_ctrl_msg::ScmSocket;

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

struct UffdPfHandler {
    mappings: Vec<RegionBackendMapping>,
    backing_buffer: *const u8,
    uffd: Uffd,
}

impl UffdPfHandler {
    pub fn from_unix_stream(stream: UnixStream, data: *const u8, size: usize) -> Self {
        let mut message_buf = vec![0u8; 1024];
        let (bytes_read, file) = stream
            .recv_with_fd(&mut message_buf[..])
            .expect("cannot recv_with_fd");
        message_buf.resize(bytes_read, 0);

        let body = String::from_utf8(message_buf).unwrap();
        println!("API response of {} bytes:\n{:?}", bytes_read, body);
        let file = file.expect("Uffd not passed through UDS!");

        let mappings =
            serde_json::from_str::<Vec<RegionBackendMapping>>(&body).expect("deser failed");
        let memsize: usize = mappings.iter().map(|r| r.size).sum();
        // Make sure memory size matches backing data size.
        assert_eq!(memsize, size);

        let uffd = unsafe { Uffd::from_raw_fd(file.into_raw_fd()) };
        Self {
            mappings,
            backing_buffer: data,
            uffd,
        }
    }

    fn serve_pf(&self, addr: *mut u8, page_size: usize) {
        let dst = (addr as usize & !(page_size as usize - 1)) as *mut c_void;
        println!(
            "    looking for file offset corresponding to hVA dst: {:?}",
            dst
        );

        let mut found_mapping = None;
        for r in self.mappings.iter() {
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
        let src = self.backing_buffer as u64 + r.offset;
        // Populate whole region from backing mem-file.
        let copy = unsafe {
            self.uffd
                .copy(src as *const _, r.base_h_va as *mut _, r.size, true)
                .expect("uffd copy")
        };
        println!("        (uffdio_copy.copy returned {})", copy);
    }

    fn run_loop(&self) {
        let page_size = sysconf(SysconfVar::PAGE_SIZE).unwrap().unwrap() as usize;
        let pollfd = PollFd::new(self.uffd.as_raw_fd(), PollFlags::POLLIN);
        println!("\nfault_handler_thread():");

        // Loop, handling incoming events on the userfaultfd file descriptor
        loop {
            println!("    waiting for PFs...");
            // See what poll() tells us about the userfaultfd
            let nready = poll(&mut [pollfd], -1).expect("poll");

            let revents = pollfd.revents().unwrap();
            println!(
                "    poll() returns: nready = {}; POLLIN = {}; POLLERR = {}",
                nready,
                revents.contains(PollFlags::POLLIN),
                revents.contains(PollFlags::POLLERR),
            );

            // Read an event from the userfaultfd
            let event = self
                .uffd
                .read_event()
                .expect("read uffd_msg")
                .expect("uffd_msg ready");

            // We expect only one kind of event; verify that assumption
            if let userfaultfd::Event::Pagefault { addr, .. } = event {
                // Display info about the page-fault event
                println!("    UFFD_EVENT_PAGEFAULT event: {:?}", event);
                self.serve_pf(addr as *mut u8, page_size);
            } else {
                panic!("Unexpected event on userfaultfd");
            }
        }
    }
}

pub fn run(uffd_sock_path: String, mem_file_path: String, barrier: Arc<Barrier>) {
    let file = File::open(mem_file_path).expect("cannot open memfile");
    let size = file.metadata().unwrap().len() as usize;
    // Create a page that will be copied into the faulting region
    let memfile_buffer = unsafe {
        mmap(
            ptr::null_mut(),
            size,
            ProtFlags::PROT_READ,
            MapFlags::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
        .expect("mmap")
    } as *const u8;

    // Get Uffd from UDS. We'll use the uffd to handle PFs for Firecracker.
    let listener = UnixListener::bind(&uffd_sock_path).expect("cannot bind");

    println!("Bound UDS at: {:?}", uffd_sock_path);
    // Signal main thread we're ready and listening on uffd UDS.
    barrier.wait();

    let (stream, _) = listener.accept().expect("cannot listen");
    let uffd_handler = UffdPfHandler::from_unix_stream(stream, memfile_buffer, size);

    uffd_handler.run_loop();
    println!("Uffd thread done!");
}
