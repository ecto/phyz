use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlCanvasElement, HtmlElement, Window};

pub fn window() -> Window {
    web_sys::window().expect("no global window")
}

pub fn document() -> Document {
    window().document().expect("no document")
}

pub fn canvas() -> HtmlCanvasElement {
    document()
        .get_element_by_id("viz")
        .expect("no #viz canvas")
        .dyn_into()
        .expect("not a canvas")
}

pub fn get_el(id: &str) -> HtmlElement {
    document()
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("no #{id}"))
        .dyn_into()
        .unwrap_or_else(|_| panic!("#{id} not HtmlElement"))
}

pub fn set_text(id: &str, text: &str) {
    if let Some(el) = document().get_element_by_id(id) {
        el.set_text_content(Some(text));
    }
}

pub fn set_class(id: &str, class: &str) {
    if let Some(el) = document().get_element_by_id(id)
        && let Ok(el) = el.dyn_into::<HtmlElement>()
    {
        el.set_class_name(class);
    }
}

pub fn set_inner_html(id: &str, html: &str) {
    if let Some(el) = document().get_element_by_id(id) {
        el.set_inner_html(html);
    }
}

pub fn set_style(id: &str, style: &str) {
    if let Some(el) = document().get_element_by_id(id) {
        el.set_attribute("style", style).ok();
    }
}

/// Detect mobile device via touch support and screen width.
pub fn is_mobile() -> bool {
    let w = window();
    // Primary signal: coarse pointer (touch device)
    let coarse = w
        .match_media("(pointer: coarse)")
        .ok()
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false);
    // Secondary signal: narrow viewport
    let narrow = w
        .inner_width()
        .ok()
        .and_then(|v| v.as_f64())
        .map(|w| w <= 768.0)
        .unwrap_or(false);
    coarse || narrow
}

pub fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("requestAnimationFrame failed");
}
