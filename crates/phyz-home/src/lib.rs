use wasm_bindgen::prelude::*;

mod app;
mod auth;
mod cache;
mod coordinator;
mod defrag;
mod dom;
mod realtime;
mod supabase;
mod viz;
mod worker;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook_set();
    wasm_bindgen_futures::spawn_local(app::run());
}

fn console_error_panic_hook_set() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        web_sys::console::error_1(&msg.into());
    }));
}
