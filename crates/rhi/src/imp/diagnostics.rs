//! Native validation diagnostic capture.

#[cfg(any(test, feature = "test-support"))]
use super::*;

#[cfg(any(test, feature = "test-support"))]
struct ValidationLogger;

#[cfg(any(test, feature = "test-support"))]
static VALIDATION_LOGGER: ValidationLogger = ValidationLogger;
#[cfg(any(test, feature = "test-support"))]
static VALIDATION_MESSAGES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
#[cfg(any(test, feature = "test-support"))]
static VALIDATION_LOGGER_INIT: std::sync::Once = std::sync::Once::new();

#[cfg(any(test, feature = "test-support"))]
impl log::Log for ValidationLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            VALIDATION_MESSAGES
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(format!("{}: {}", record.target(), record.args()));
        }
    }
    fn flush(&self) {}
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn initialize_validation_capture() {
    VALIDATION_LOGGER_INIT.call_once(|| {
        let _ = log::set_logger(&VALIDATION_LOGGER);
        // wgpu-hal chooses Vulkan debug-utils severities from this global
        // filter while creating the instance, so Warn must be enabled before
        // Device::open to capture validation warnings as well as errors.
        log::set_max_level(log::LevelFilter::Warn);
    });
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn clear_validation_diagnostics(owner: &Arc<OpenedDevice>) {
    VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clear();
    #[cfg(feature = "dx12")]
    if let NativeDevice::Dx12 { device, .. } = &owner.native {
        use windows::{Win32::Graphics::Direct3D12::ID3D12InfoQueue, core::Interface as _};
        if let Ok(queue) = device.raw_device().cast::<ID3D12InfoQueue>() {
            // SAFETY: the information queue belongs to the retained live device.
            unsafe { queue.ClearStoredMessages() };
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub(crate) fn validation_diagnostics(owner: &Arc<OpenedDevice>) -> Vec<String> {
    let mut messages = VALIDATION_MESSAGES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .clone();
    #[cfg(feature = "dx12")]
    if let NativeDevice::Dx12 { device, .. } = &owner.native {
        use windows::{
            Win32::Graphics::Direct3D12::{
                D3D12_MESSAGE, D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                D3D12_MESSAGE_SEVERITY_CORRUPTION, D3D12_MESSAGE_SEVERITY_ERROR,
                D3D12_MESSAGE_SEVERITY_WARNING, ID3D12InfoQueue,
            },
            core::Interface as _,
        };
        if let Ok(queue) = device.raw_device().cast::<ID3D12InfoQueue>() {
            // SAFETY: read-only count query on the retained device queue.
            let count = unsafe { queue.GetNumStoredMessagesAllowedByRetrievalFilter() };
            for index in 0..count {
                let mut byte_length = 0;
                // SAFETY: a null message pointer requests the byte size for the
                // stored message; `byte_length` is a valid out-pointer.
                if unsafe { queue.GetMessage(index, None, &mut byte_length) }.is_err()
                    || byte_length < core::mem::size_of::<D3D12_MESSAGE>()
                {
                    messages.push(format!(
                        "DX12 InfoQueue message {index} could not be retrieved (size={byte_length})"
                    ));
                    continue;
                }
                let words = byte_length.div_ceil(core::mem::size_of::<usize>());
                // `usize` storage supplies the alignment required by D3D12_MESSAGE.
                let mut storage = vec![0_usize; words];
                let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
                // SAFETY: storage has the exact queried byte capacity and is
                // sufficiently aligned; InfoQueue writes one POD message plus
                // its null-terminated description into that buffer.
                if unsafe { queue.GetMessage(index, Some(message), &mut byte_length) }.is_err() {
                    messages.push(format!("DX12 InfoQueue message {index} could not be read"));
                    continue;
                }
                // SAFETY: GetMessage initialized a D3D12_MESSAGE at `message`.
                let message = unsafe { &*message };
                let description = if message.pDescription.is_null() {
                    String::new()
                } else {
                    // SAFETY: D3D12_MESSAGE descriptions are null-terminated
                    // ANSI strings within the queried message allocation.
                    unsafe {
                        std::ffi::CStr::from_ptr(message.pDescription.cast())
                            .to_string_lossy()
                            .into_owned()
                    }
                };
                let rendered = format!(
                    "DX12 InfoQueue severity={:?} category={:?} id={:?}: {description}",
                    message.Severity, message.Category, message.ID,
                );
                if message.ID == D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE
                    || message.ID == D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE
                {
                    // wgpu-hal's portable TextureDescriptor exposes no D3D12
                    // optimized-clear value, and upstream wgpu-hal 30 filters
                    // color/depth clear-value mismatch messages (#820/#821) as
                    // non-actionable performance spam. The clear is still
                    // semantically correct, so keep it observable without
                    // classifying it as a correctness diagnostic.
                    eprintln!("ignored {rendered}");
                    continue;
                }
                let blocking = matches!(
                    message.Severity,
                    D3D12_MESSAGE_SEVERITY_CORRUPTION
                        | D3D12_MESSAGE_SEVERITY_ERROR
                        | D3D12_MESSAGE_SEVERITY_WARNING
                );
                if blocking {
                    messages.push(rendered);
                } else {
                    // Required validation gates report only Warning/Error (and
                    // Corruption) diagnostics, matching Vulkan's Warn capture.
                    // Keep lower-severity messages observable in hardware logs
                    // without turning normal driver info into a false failure.
                    eprintln!("ignored {rendered}");
                }
            }
        }
    }
    messages
}
