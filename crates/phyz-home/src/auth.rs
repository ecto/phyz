use serde::{Deserialize, Serialize};

/// Authenticated user session from Supabase magic link.
#[derive(Clone, Serialize, Deserialize)]
pub struct AuthSession {
    pub access_token: String,
    pub user_id: String,
    pub email: String,
}

/// Check URL fragment for magic link callback tokens.
/// Supabase redirects to `app/#access_token=xxx&type=magiclink&...`
pub fn check_callback() -> Option<AuthSession> {
    let window = crate::dom::window();
    let href = window.location().href().ok()?;
    let url = web_sys::Url::new(&href).ok()?;
    let hash = url.hash();
    if hash.is_empty() {
        return None;
    }

    // Hash looks like "#access_token=xxx&token_type=bearer&..."
    // Parse as query params by replacing leading # with ?
    let fake_url = format!("https://x?{}", &hash[1..]);
    let parsed = web_sys::Url::new(&fake_url).ok()?;
    let params = parsed.search_params();

    let access_token = params.get("access_token")?;
    let token_type = params.get("type").unwrap_or_default();
    if token_type != "magiclink" && token_type != "recovery" {
        // Only handle magic link callbacks
        if access_token.is_empty() {
            return None;
        }
    }

    // Decode the JWT to extract user_id and email from payload
    let (user_id, email) = decode_jwt_claims(&access_token)?;

    // Clear the hash fragment
    clear_hash();

    let session = AuthSession {
        access_token,
        user_id,
        email,
    };

    // Persist session
    crate::cache::save_session(&session);

    Some(session)
}

/// Decode JWT payload (base64url middle segment) to extract sub and email.
fn decode_jwt_claims(jwt: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return None;
    }

    // base64url decode the payload
    let payload = base64url_decode(parts[1])?;
    let json: serde_json::Value = serde_json::from_str(&payload).ok()?;

    let sub = json.get("sub")?.as_str()?.to_string();
    let email = json.get("email")?.as_str().unwrap_or("").to_string();

    Some((sub, email))
}

/// Minimal base64url decoder (no padding required).
fn base64url_decode(input: &str) -> Option<String> {
    const TABLE: &[u8; 128] = &{
        let mut t = [255u8; 128];
        let mut i = 0u8;
        while i < 26 {
            t[(b'A' + i) as usize] = i;
            t[(b'a' + i) as usize] = i + 26;
            i += 1;
        }
        let mut d = 0u8;
        while d < 10 {
            t[(b'0' + d) as usize] = d + 52;
            d += 1;
        }
        // standard base64
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        // base64url
        t[b'-' as usize] = 62;
        t[b'_' as usize] = 63;
        t
    };

    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &b in bytes {
        if b == b'=' {
            break;
        }
        let val = if (b as usize) < 128 { TABLE[b as usize] } else { 255 };
        if val == 255 {
            continue;
        }
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    String::from_utf8(out).ok()
}

/// Clear URL hash fragment after parsing callback.
fn clear_hash() {
    let window = crate::dom::window();
    if let Ok(history) = window.history() {
        let location = window.location();
        let path = location.pathname().unwrap_or_default();
        let search = location.search().unwrap_or_default();
        history
            .replace_state_with_url(
                &wasm_bindgen::JsValue::NULL,
                "",
                Some(&format!("{path}{search}")),
            )
            .ok();
    }
}

/// Load existing session from localStorage, clearing it if the JWT is expired.
pub fn load_session() -> Option<AuthSession> {
    let session = crate::cache::load_session()?;
    if is_jwt_expired(&session.access_token) {
        web_sys::console::log_1(&"auth: session expired, clearing".into());
        crate::cache::clear_session();
        return None;
    }
    Some(session)
}

/// Check if a JWT's `exp` claim is in the past.
fn is_jwt_expired(jwt: &str) -> bool {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return true;
    }
    let Some(payload) = base64url_decode(parts[1]) else {
        return true;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&payload) else {
        return true;
    };
    let Some(exp) = json.get("exp").and_then(|v| v.as_f64()) else {
        return true;
    };
    let now = js_sys::Date::now() / 1000.0;
    now >= exp
}

/// Sign out: clear session from localStorage.
pub fn sign_out() {
    crate::cache::clear_session();
}
