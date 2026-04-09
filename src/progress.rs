use std::io::{IsTerminal, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

const FRAMES: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const TICK_MS: u64 = 80;

pub(crate) struct Spinner {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Spinner {
    pub(crate) fn new(msg: &str) -> Self {
        let done = Arc::new(AtomicBool::new(false));

        let thread = if std::io::stderr().is_terminal() {
            let done = Arc::clone(&done);
            let msg = msg.to_string();
            Some(thread::spawn(move || {
                let mut err = std::io::stderr();
                let mut i = 0;
                loop {
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    let _ = write!(err, "\r\x1b[2K{} {}", FRAMES[i % FRAMES.len()], msg);
                    let _ = err.flush();
                    thread::sleep(Duration::from_millis(TICK_MS));
                    i += 1;
                }
            }))
        } else {
            None
        };

        Self { done, thread }
    }

    pub(crate) fn finish(self, msg: &str) {
        let is_tty = self.thread.is_some();
        drop(self);
        if is_tty {
            eprintln!("\x1b[32m✓\x1b[0m {msg}");
        }
    }

    pub(crate) fn cancel(self) {
        tracing::trace!("spinner cancelled");
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
            eprint!("\r\x1b[2K");
            let _ = std::io::stderr().flush();
        }
    }
}
