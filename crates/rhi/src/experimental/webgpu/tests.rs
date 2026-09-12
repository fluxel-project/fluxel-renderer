//! Unit contracts for the closed browser executor types.

use js_sys::{Function, Promise};
use wasm_bindgen_test::*;

use super::*;
use crate::experimental::webgpu::js::BrowserRequestProvider;

wasm_bindgen_test_configure!(run_in_browser);

struct PendingRequest {
    target: JsValue,
    kind: RequestKind,
    resolve: Function,
    reject: Function,
}

#[derive(Clone, Copy)]
enum RequestKind {
    Adapter,
    Device,
}

struct GatedBrowserRequests {
    production: ProductionBrowserRequests,
    defer_adapter: Cell<bool>,
    defer_device: Cell<bool>,
    adapter_calls: Cell<usize>,
    device_calls: Cell<usize>,
    adapter_pending: Rc<RefCell<Option<PendingRequest>>>,
    device_pending: Rc<RefCell<Option<PendingRequest>>>,
}

impl Default for GatedBrowserRequests {
    fn default() -> Self {
        Self {
            production: ProductionBrowserRequests,
            defer_adapter: Cell::new(false),
            defer_device: Cell::new(false),
            adapter_calls: Cell::new(0),
            device_calls: Cell::new(0),
            adapter_pending: Rc::new(RefCell::new(None)),
            device_pending: Rc::new(RefCell::new(None)),
        }
    }
}

impl GatedBrowserRequests {
    fn arm_adapter(&self) {
        self.defer_adapter.set(true);
    }

    fn arm_device(&self) {
        self.defer_device.set(true);
    }

    fn pending_adapter(&self) -> bool {
        self.adapter_pending.borrow().is_some()
    }

    fn pending_device(&self) -> bool {
        self.device_pending.borrow().is_some()
    }

    fn release_adapter(&self) {
        Self::release(&self.adapter_pending);
    }

    fn release_device(&self) {
        Self::release(&self.device_pending);
    }

    fn reject_adapter(&self, message: &str) {
        Self::reject(&self.adapter_pending, message);
    }

    fn reject_device(&self, message: &str) {
        Self::reject(&self.device_pending, message);
    }

    fn deferred(
        target: &JsValue,
        kind: RequestKind,
        pending: Rc<RefCell<Option<PendingRequest>>>,
    ) -> Promise {
        let target = target.clone();
        Promise::new(&mut move |resolve, reject| {
            assert!(
                pending
                    .borrow_mut()
                    .replace(PendingRequest {
                        target: target.clone(),
                        kind,
                        resolve,
                        reject,
                    })
                    .is_none(),
                "only one browser request may occupy a gate"
            );
        })
    }

    fn release(pending: &RefCell<Option<PendingRequest>>) {
        let pending = pending
            .borrow_mut()
            .take()
            .expect("a deferred browser request is pending");
        let production = ProductionBrowserRequests;
        let native = match pending.kind {
            RequestKind::Adapter => production.request_adapter(&pending.target),
            RequestKind::Device => production.request_device(&pending.target),
        };
        match native {
            Ok(native) => {
                let _ = call2(
                    &native,
                    "then",
                    pending.resolve.as_ref(),
                    pending.reject.as_ref(),
                )
                .expect("native browser request exposes Promise.then");
            }
            Err(error) => {
                pending
                    .reject
                    .call1(&JsValue::UNDEFINED, &error)
                    .expect("reject deferred browser request");
            }
        }
    }

    fn reject(pending: &RefCell<Option<PendingRequest>>, message: &str) {
        let pending = pending
            .borrow_mut()
            .take()
            .expect("a deferred browser request is pending");
        pending
            .reject
            .call1(&JsValue::UNDEFINED, &js_sys::Error::new(message))
            .expect("reject deferred browser request");
    }
}

impl BrowserRequestProvider for GatedBrowserRequests {
    fn request_adapter(&self, gpu: &JsValue) -> Result<Promise, JsValue> {
        self.adapter_calls.set(self.adapter_calls.get() + 1);
        if self.defer_adapter.replace(false) {
            Ok(Self::deferred(
                gpu,
                RequestKind::Adapter,
                Rc::clone(&self.adapter_pending),
            ))
        } else {
            self.production.request_adapter(gpu)
        }
    }

    fn request_device(&self, adapter: &JsValue) -> Result<Promise, JsValue> {
        self.device_calls.set(self.device_calls.get() + 1);
        if self.defer_device.replace(false) {
            Ok(Self::deferred(
                adapter,
                RequestKind::Device,
                Rc::clone(&self.device_pending),
            ))
        } else {
            self.production.request_device(adapter)
        }
    }
}

fn canvas() -> JsValue {
    js_sys::eval("document.createElement('canvas')")
        .expect("browser test creates an isolated canvas")
}

async fn turn() {
    let promise = js_sys::eval("new Promise(resolve => setTimeout(resolve, 0))")
        .expect("browser test can yield to the event loop");
    JsFuture::from(Promise::from(promise))
        .await
        .expect("event-loop turn resolves");
}

async fn wait_until(mut predicate: impl FnMut() -> bool, description: &str) {
    for _ in 0..200 {
        if predicate() {
            return;
        }
        turn().await;
    }
    panic!("timed out waiting for {description}");
}

async fn lose(session: &mut WebGpuSession) {
    session
        .controlled_destroy_for_evidence()
        .expect("destroy current real device");
    wait_until(
        || session.state() == WebGpuSessionState::Lost,
        "the production device.lost callback",
    )
    .await;
}

async fn dispose(session: &mut WebGpuSession) {
    let promise = session.dispose().expect("start terminal disposal");
    JsFuture::from(promise)
        .await
        .expect("terminal disposal completes");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
}

#[test]
fn formats_are_closed() {
    assert_eq!(
        WebGpuCanvasFormat::parse("rgba8unorm"),
        Some(WebGpuCanvasFormat::Rgba8Unorm)
    );
    assert!(WebGpuCanvasFormat::parse("rgba8unorm-srgb").is_none());
}

#[test]
fn lifecycle_has_distinct_loss_and_dispose() {
    assert_ne!(WebGpuSessionState::Lost, WebGpuSessionState::Disposed);
    assert_ne!(MAX_FRAMES_IN_FLIGHT, 0);
}

#[test]
fn recovery_publication_requires_its_original_token_and_state() {
    let mut shared = Shared {
        state: WebGpuSessionState::Recovering,
        generation: 7,
        token: 11,
        ..Shared::default()
    };
    assert!(recovery_attempt_current(shared.state, shared.token, 11));

    // `dispose` invalidates the token before it waits for the in-flight
    // recovery. The candidate must therefore never publish its facts.
    shared.token += 1;
    shared.state = WebGpuSessionState::Disposing;
    assert!(!recovery_attempt_current(shared.state, shared.token, 11));
    assert_eq!(shared.generation, 7);
}

#[test]
fn recovery_candidate_is_not_committable_from_a_terminal_state() {
    let shared = Shared {
        state: WebGpuSessionState::Disposed,
        token: 3,
        ..Shared::default()
    };
    assert!(!recovery_attempt_current(shared.state, shared.token, 3));
}

#[wasm_bindgen_test(async)]
async fn public_constructor_uses_the_production_browser_provider() {
    let mut session = WebGpuSession::new(canvas())
        .await
        .expect("the public constructor opens a real WebGPU session");
    assert_eq!(session.generation(), 1);
    assert_eq!(session.state(), WebGpuSessionState::Active);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_resolves_the_real_adapter_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    assert_eq!(requests.adapter_calls.get(), 1);
    assert_eq!(requests.device_calls.get(), 1);
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    assert_eq!(session.state(), WebGpuSessionState::Recovering);
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    assert_eq!(session.generation(), 1);
    requests.release_adapter();
    JsFuture::from(recovery)
        .await
        .expect("native adapter and device requests resolve");

    assert_eq!(session.state(), WebGpuSessionState::Active);
    assert_eq!(session.generation(), 2);
    assert_eq!(requests.adapter_calls.get(), 2);
    assert_eq!(requests.device_calls.get(), 2);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_rejects_the_adapter_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    requests.reject_adapter("injected requestAdapter rejection");
    JsFuture::from(recovery)
        .await
        .expect_err("recovery must reject");

    assert_eq!(session.state(), WebGpuSessionState::Poisoned);
    assert_eq!(session.generation(), 1);
    assert!(session.diagnostics().iter().any(|diagnostic| {
        diagnostic.code == "recovery-request-failed"
            && diagnostic.operation == "recover-request"
            && diagnostic.generation == 1
    }));
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_resolves_the_real_device_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_device();
    let recovery = session.recover().expect("start recovery");
    wait_until(|| requests.pending_device(), "deferred requestDevice call").await;
    assert_eq!(session.state(), WebGpuSessionState::Recovering);
    assert_eq!(session.generation(), 1);
    requests.release_device();
    JsFuture::from(recovery)
        .await
        .expect("native device request resolves");

    assert_eq!(session.state(), WebGpuSessionState::Active);
    assert_eq!(session.generation(), 2);
    assert_eq!(requests.adapter_calls.get(), 2);
    assert_eq!(requests.device_calls.get(), 2);
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn recovery_defers_then_rejects_the_device_request() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_device();
    let recovery = session.recover().expect("start recovery");
    wait_until(|| requests.pending_device(), "deferred requestDevice call").await;
    requests.reject_device("injected requestDevice rejection");
    JsFuture::from(recovery)
        .await
        .expect_err("recovery must reject");

    assert_eq!(session.state(), WebGpuSessionState::Poisoned);
    assert_eq!(session.generation(), 1);
    assert!(session.diagnostics().iter().any(|diagnostic| {
        diagnostic.code == "recovery-request-failed"
            && diagnostic.operation == "recover-request"
            && diagnostic.generation == 1
    }));
    dispose(&mut session).await;
}

#[wasm_bindgen_test(async)]
async fn disposal_joins_a_deferred_adapter_resolution_without_publication() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    let disposal = session.dispose().expect("dispose pending recovery");
    assert_eq!(session.state(), WebGpuSessionState::Disposing);
    requests.release_adapter();

    JsFuture::from(recovery)
        .await
        .expect("stale recovery resolves without publication");
    JsFuture::from(disposal)
        .await
        .expect("disposal joins stale native requests");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
    assert_eq!(session.generation(), 1);
    assert!(
        session
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.code != "recovery-request-failed")
    );
}

#[wasm_bindgen_test(async)]
async fn disposal_joins_a_deferred_adapter_rejection_without_poisoning() {
    let requests = Rc::new(GatedBrowserRequests::default());
    let mut session = WebGpuSession::new_with_requests(canvas(), requests.clone())
        .await
        .expect("open a real WebGPU session");
    lose(&mut session).await;

    requests.arm_adapter();
    let recovery = session.recover().expect("start recovery");
    wait_until(
        || requests.pending_adapter(),
        "deferred requestAdapter call",
    )
    .await;
    let disposal = session.dispose().expect("dispose pending recovery");
    assert_eq!(session.state(), WebGpuSessionState::Disposing);
    requests.reject_adapter("stale requestAdapter rejection");

    JsFuture::from(recovery)
        .await
        .expect("stale rejection does not reject recovery");
    JsFuture::from(disposal)
        .await
        .expect("disposal joins stale rejection");
    assert_eq!(session.state(), WebGpuSessionState::Disposed);
    assert_eq!(session.generation(), 1);
    assert!(
        session
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.code != "recovery-request-failed")
    );
}
