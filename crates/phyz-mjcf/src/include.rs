//! `<include file="..."/>` expansion.
//!
//! MJCF `<include>` splices another file's contents in at that point in the
//! tree. Menagerie ships its robots as `scene.xml` files that are little more
//! than an `<include>` plus a floor, and the dm_control suite pulls shared
//! visual settings the same way — without this, those files parse to an empty
//! model rather than failing, which is the worst of both worlds.
//!
//! Expansion happens as a source-to-source pass before the parser runs, so the
//! parser itself never has to know includes exist.

use crate::{MjcfError, Result};
use quick_xml::events::{BytesStart, Event};
use quick_xml::{Reader, Writer};
use std::io::Cursor;
use std::path::{Path, PathBuf};

/// Cap on nesting depth, which also breaks include cycles.
const MAX_DEPTH: usize = 16;

/// Recursively expand every `<include>` in `xml`, resolving paths against `dir`.
///
/// Returns the document unchanged when it contains no includes, so the common
/// case costs one scan and no re-serialisation.
pub fn expand(xml: &str, dir: Option<&Path>) -> Result<String> {
    if !has_include(xml) {
        return Ok(xml.to_string());
    }
    let mut out = Writer::new(Cursor::new(Vec::new()));
    expand_into(xml, dir, 0, &mut out, false)?;
    let bytes = out.into_inner().into_inner();
    String::from_utf8(bytes).map_err(|e| {
        MjcfError::InvalidMjcf(format!("include expansion produced invalid UTF-8: {e}"))
    })
}

/// Cheap pre-check so include-free documents skip the rewrite entirely.
fn has_include(xml: &str) -> bool {
    xml.contains("<include")
}

fn expand_into(
    xml: &str,
    dir: Option<&Path>,
    depth: usize,
    out: &mut Writer<Cursor<Vec<u8>>>,
    // Included files are wrapped in <mujoco>/<mujocoinclude>; that wrapper is
    // not part of the splice, only its children are.
    strip_root: bool,
) -> Result<()> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);
    let mut buf = Vec::new();
    let mut nesting = 0usize;

    loop {
        let event = reader
            .read_event_into(&mut buf)
            .map_err(MjcfError::XmlError)?;
        match event {
            Event::Eof => break,
            Event::Start(ref e) => {
                nesting += 1;
                if strip_root && nesting == 1 {
                    buf.clear();
                    continue;
                }
                if is_include(e) {
                    // A <include> with children is malformed; treat it as the
                    // splice point and skip to its end.
                    splice(e, dir, depth, out)?;
                    buf.clear();
                    continue;
                }
                out.write_event(Event::Start(e.clone())).map_err(io_err)?;
            }
            Event::End(ref e) => {
                if strip_root && nesting == 1 {
                    nesting -= 1;
                    buf.clear();
                    continue;
                }
                nesting = nesting.saturating_sub(1);
                out.write_event(Event::End(e.clone())).map_err(io_err)?;
            }
            Event::Empty(ref e) => {
                if is_include(e) {
                    splice(e, dir, depth, out)?;
                    buf.clear();
                    continue;
                }
                out.write_event(Event::Empty(e.clone())).map_err(io_err)?;
            }
            other => {
                out.write_event(other).map_err(io_err)?;
            }
        }
        buf.clear();
    }
    Ok(())
}

fn is_include(e: &BytesStart) -> bool {
    e.name().as_ref() == b"include"
}

/// Read the file named by an `<include>` and write its children into `out`.
fn splice(
    e: &BytesStart,
    dir: Option<&Path>,
    depth: usize,
    out: &mut Writer<Cursor<Vec<u8>>>,
) -> Result<()> {
    let file = include_file(e)?;
    if depth + 1 >= MAX_DEPTH {
        return Err(MjcfError::InvalidMjcf(format!(
            "<include> nesting exceeded {MAX_DEPTH} levels at '{file}'; \
             the includes are probably cyclic"
        )));
    }
    let path = match dir {
        Some(d) => d.join(&file),
        None => PathBuf::from(&file),
    };
    let content = std::fs::read_to_string(&path).map_err(|err| {
        MjcfError::InvalidMjcf(format!(
            "<include file=\"{file}\"> could not be read: {err}"
        ))
    })?;
    let nested_dir = path.parent().map(Path::to_path_buf);
    expand_into(&content, nested_dir.as_deref(), depth + 1, out, true)
}

fn include_file(e: &BytesStart) -> Result<String> {
    for attr in e.attributes() {
        let attr = attr.map_err(|err| MjcfError::InvalidMjcf(err.to_string()))?;
        if attr.key.as_ref() == b"file" {
            return Ok(String::from_utf8_lossy(&attr.value).to_string());
        }
    }
    Err(MjcfError::MissingAttribute {
        element: "include".to_string(),
        attribute: "file".to_string(),
    })
}

fn io_err(e: std::io::Error) -> MjcfError {
    MjcfError::IoError(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_without_includes_is_returned_verbatim() {
        let xml = "<mujoco><worldbody/></mujoco>";
        assert_eq!(expand(xml, None).unwrap(), xml);
    }

    #[test]
    fn missing_file_attribute_is_an_error() {
        let err = expand("<mujoco><include/></mujoco>", None).unwrap_err();
        assert!(matches!(err, MjcfError::MissingAttribute { .. }), "{err}");
    }

    #[test]
    fn unreadable_include_names_the_file() {
        let err = expand(r#"<mujoco><include file="nope.xml"/></mujoco>"#, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope.xml"), "{err}");
    }
}
