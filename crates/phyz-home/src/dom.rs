use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
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
    if let Some(el) = document().get_element_by_id(id) {
        if let Ok(el) = el.dyn_into::<HtmlElement>() {
            el.set_class_name(class);
        }
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

/// Append structured log entries to the #log element.
pub fn append_log(entries: &[String]) {
    let Some(el) = document().get_element_by_id("log") else {
        return;
    };

    // Remove "waiting" message on first entry
    if el.child_element_count() == 1 {
        if let Some(first) = el.first_element_child() {
            if first.class_name() == "empty" {
                el.remove_child(&first).ok();
            }
        }
    }

    for entry in entries {
        let div = document().create_element("div").unwrap();
        div.set_class_name("entry fresh");
        // Entry is pre-formatted HTML
        div.set_inner_html(entry);
        el.append_child(&div).ok();
    }

    // Keep only the last 6 entries
    while el.child_element_count() > 6 {
        if let Some(first) = el.first_child() {
            el.remove_child(&first).ok();
        }
    }

    // Scroll to bottom
    el.set_scroll_top(el.scroll_height());
}

pub fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    window()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("requestAnimationFrame failed");
}
