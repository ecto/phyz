use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, Request, RequestInit, RequestMode, Response};

const SUPABASE_URL: &str = "https://ynhqxopghmhkeixgbfqi.supabase.co";
const SUPABASE_ANON_KEY: &str = "sb_publishable_i-0RH4wZsFMaLNMXt2UCCg_ZqDoLmJr";
const PROJECT_REF: &str = "ynhqxopghmhkeixgbfqi";

pub fn anon_key() -> &'static str {
    SUPABASE_ANON_KEY
}

pub fn project_ref() -> &'static str {
    PROJECT_REF
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUnit {
    pub id: String,
    pub params: WorkParams,
    pub completed_count: i32,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkParams {
    pub level: u32,
    pub coupling_g2: f64,
    pub geometry_seed: u64,
    pub perturbation: Perturbation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Perturbation {
    #[serde(rename = "base")]
    Base,
    #[serde(rename = "edge")]
    Edge { index: usize, direction: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultPayload {
    pub ground_state_energy: f64,
    pub entropy_per_partition: Vec<f64>,
    #[serde(default)]
    pub boundary_area_per_partition: Vec<f64>,
    pub walltime_ms: f64,
}

/// A computed result waiting to be batch-submitted.
#[derive(Debug, Clone, Serialize)]
pub struct PendingResult {
    pub work_unit_id: String,
    pub contributor_id: String,
    pub result: ResultPayload,
}

/// A result from the global feed with contributor name and work unit context.
#[derive(Debug, Clone, Deserialize)]
pub struct FeedResult {
    pub result_id: String,
    pub contributor_id: String,
    pub contributor_name: String,
    pub coupling_g2: f64,
    pub result_data: ResultPayload,
    pub submitted_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LeaderboardEntry {
    #[serde(rename = "name")]
    pub name: Option<String>,
    pub units: i64,
    pub player_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContributorStats {
    pub total_units: i64,
    pub display_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ExperimentProgress {
    pub submitted: usize,
    pub consensus: usize,
    pub partial: usize,
    pub total: usize,
}

pub struct SupabaseClient {
    url: String,
    key: String,
}

impl SupabaseClient {
    pub fn new() -> Self {
        Self {
            url: SUPABASE_URL.to_string(),
            key: SUPABASE_ANON_KEY.to_string(),
        }
    }

    fn api_url(&self, path: &str) -> String {
        format!("{}/rest/v1/{}", self.url, path)
    }

    fn headers(&self) -> Result<Headers, JsValue> {
        self.headers_with_auth(None)
    }

    fn headers_with_auth(&self, jwt: Option<&str>) -> Result<Headers, JsValue> {
        let headers = Headers::new()?;
        headers.set("apikey", &self.key)?;
        let bearer = jwt.unwrap_or(&self.key);
        headers.set("Authorization", &format!("Bearer {bearer}"))?;
        headers.set("Content-Type", "application/json")?;
        headers.set("Prefer", "return=minimal")?;
        Ok(headers)
    }

    /// Fetch pending work units (up to `limit`) in random order, prioritizing L1 over L0.
    /// Uses a server-side RPC function so different volunteers get different subsets.
    pub async fn fetch_pending_work(&self, limit: usize) -> Result<Vec<WorkUnit>, String> {
        let url = format!(
            "{}/rest/v1/rpc/fetch_pending_work",
            self.url,
        );
        let body = serde_json::json!({ "p_limit": limit }).to_string();
        let resp = self.post(&url, &body).await?;
        let text = self.text(resp).await?;
        serde_json::from_str(&text).map_err(|e| format!("parse error: {e}"))
    }

    /// Fetch recent results with contributor names for the live viz feed.
    /// Pass `after` to get only results newer than a timestamp (for incremental updates).
    pub async fn fetch_results_feed(
        &self,
        limit: usize,
        after: Option<&str>,
    ) -> Result<Vec<FeedResult>, String> {
        let url = format!("{}/rest/v1/rpc/recent_results_feed", self.url);
        let mut body = serde_json::json!({ "p_limit": limit });
        if let Some(ts) = after {
            body["p_after"] = serde_json::Value::String(ts.to_string());
        }
        let resp = self.post(&url, &body.to_string()).await?;
        let text = self.text(resp).await?;
        serde_json::from_str(&text).map_err(|e| format!("parse feed: {e}"))
    }

    /// Register or update a contributor.
    pub async fn upsert_contributor(&self, fingerprint: &str) -> Result<String, String> {
        let body = serde_json::json!({
            "fingerprint": fingerprint,
        });

        let url = format!(
            "{}?on_conflict=fingerprint",
            self.api_url("contributors"),
        );

        let headers = self.headers().map_err(|e| format!("{e:?}"))?;
        headers.set("Prefer", "return=representation,resolution=merge-duplicates").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body.to_string()));

        let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;
        let resp = self.do_fetch(request).await?;
        let text = self.text(resp).await?;

        #[derive(Deserialize)]
        struct Row {
            id: String,
        }
        let rows: Vec<Row> = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;
        rows.first()
            .map(|r| r.id.clone())
            .ok_or_else(|| "no contributor returned".to_string())
    }

    /// Get total contributor count.
    pub async fn contributor_count(&self) -> Result<usize, String> {
        let url = format!("{}?select=id&total_units=gt.0", self.api_url("contributors"));
        let headers = self.headers().map_err(|e| format!("{e:?}"))?;
        headers.set("Prefer", "count=exact").map_err(|e| format!("{e:?}"))?;
        headers.set("Range", "0-0").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;
        let resp = self.do_fetch(request).await?;

        let headers = resp.headers();
        if let Ok(Some(range)) = headers.get("content-range") {
            if let Some(n) = range.split('/').nth(1) {
                return n.parse().map_err(|e| format!("parse count: {e}"));
            }
        }
        Ok(0)
    }

    /// Fetch experiment-wide progress: (results submitted, consensus complete, total work units).
    pub async fn experiment_progress(&self) -> Result<ExperimentProgress, String> {
        let total = self.count_rows(&format!("{}?select=id", self.api_url("work_units"))).await?;
        let submitted = self.count_rows(&format!("{}?select=id", self.api_url("results"))).await?;
        let consensus = self.count_rows(&format!("{}?select=id&status=eq.complete", self.api_url("work_units"))).await?;
        let partial = self.count_rows(&format!("{}?select=id&completed_count=gt.0&status=neq.complete", self.api_url("work_units"))).await?;
        Ok(ExperimentProgress { submitted, consensus, partial, total })
    }

    /// Count rows using PostgREST `Prefer: count=exact` + `Range: 0-0`.
    async fn count_rows(&self, url: &str) -> Result<usize, String> {
        let headers = self.headers().map_err(|e| format!("{e:?}"))?;
        headers.set("Prefer", "count=exact").map_err(|e| format!("{e:?}"))?;
        headers.set("Range", "0-0").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts).map_err(|e| format!("{e:?}"))?;
        let resp = self.do_fetch(request).await?;

        let headers = resp.headers();
        if let Ok(Some(range)) = headers.get("content-range") {
            if let Some(n) = range.split('/').nth(1) {
                return n.parse().map_err(|e| format!("parse count: {e}"));
            }
        }
        Ok(0)
    }

    /// Submit a batch of results in one POST (PostgREST array insert).
    pub async fn submit_results_batch(&self, results: &[PendingResult]) -> Result<(), String> {
        if results.is_empty() {
            return Ok(());
        }
        let body = serde_json::to_string(results).map_err(|e| format!("serialize: {e}"))?;
        let url = self.api_url("results");
        self.post(&url, &body).await?;
        Ok(())
    }

    /// Send a magic link email via Supabase Auth.
    pub async fn send_magic_link(&self, email: &str) -> Result<(), String> {
        let url = format!("{}/auth/v1/magiclink", self.url);
        let body = serde_json::json!({ "email": email });

        let headers = Headers::new().map_err(|e| format!("{e:?}"))?;
        headers.set("apikey", &self.key).map_err(|e| format!("{e:?}"))?;
        headers.set("Content-Type", "application/json").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body.to_string()));

        let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;
        let resp = self.do_fetch(request).await?;

        // 200 = sent, anything else is an error
        if !resp.ok() {
            let text = self.text(resp).await.unwrap_or_default();
            return Err(format!("magic link failed: {text}"));
        }
        Ok(())
    }

    /// Fetch the leaderboard view.
    pub async fn fetch_leaderboard(&self) -> Result<Vec<LeaderboardEntry>, String> {
        let url = format!(
            "{}?select=player_id,name,units&order=units.desc&limit=50",
            self.api_url("leaderboard"),
        );
        let resp = self.get(&url).await?;
        let text = self.text(resp).await?;
        serde_json::from_str(&text).map_err(|e| format!("parse leaderboard: {e}"))
    }

    /// Link an auth account to a contributor row.
    pub async fn link_auth(
        &self,
        contributor_id: &str,
        auth_id: &str,
        jwt: &str,
    ) -> Result<(), String> {
        let url = format!(
            "{}?id=eq.{}",
            self.api_url("contributors"),
            contributor_id,
        );
        let body = serde_json::json!({
            "auth_id": auth_id,
        });

        let headers = self.headers_with_auth(Some(jwt)).map_err(|e| format!("{e:?}"))?;
        headers.set("Prefer", "return=minimal").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("PATCH");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body.to_string()));

        let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;
        self.do_fetch(request).await?;
        Ok(())
    }

    /// Set or update display name for a contributor.
    pub async fn set_display_name(
        &self,
        contributor_id: &str,
        name: &str,
        jwt: &str,
    ) -> Result<(), String> {
        let url = format!(
            "{}?id=eq.{}",
            self.api_url("contributors"),
            contributor_id,
        );
        let body = serde_json::json!({ "display_name": name });

        let headers = self.headers_with_auth(Some(jwt)).map_err(|e| format!("{e:?}"))?;
        headers.set("Prefer", "return=minimal").map_err(|e| format!("{e:?}"))?;

        let opts = RequestInit::new();
        opts.set_method("PATCH");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body.to_string()));

        let request = Request::new_with_str_and_init(&url, &opts).map_err(|e| format!("{e:?}"))?;
        self.do_fetch(request).await?;
        Ok(())
    }

    /// Fetch server-enforced contributor stats.
    pub async fn fetch_my_stats(&self, contributor_id: &str) -> Result<ContributorStats, String> {
        let url = format!(
            "{}?id=eq.{}&select=total_units,display_name",
            self.api_url("contributors"),
            contributor_id,
        );
        let resp = self.get(&url).await?;
        let text = self.text(resp).await?;
        let rows: Vec<ContributorStats> =
            serde_json::from_str(&text).map_err(|e| format!("parse stats: {e}"))?;
        rows.into_iter()
            .next()
            .ok_or_else(|| "contributor not found".to_string())
    }

    async fn get(&self, url: &str) -> Result<Response, String> {
        let headers = self.headers().map_err(|e| format!("{e:?}"))?;
        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(url, &opts).map_err(|e| format!("{e:?}"))?;
        self.do_fetch(request).await
    }

    async fn post(&self, url: &str, body: &str) -> Result<Response, String> {
        let headers = self.headers().map_err(|e| format!("{e:?}"))?;
        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_headers(&headers.into());
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(body));

        let request = Request::new_with_str_and_init(url, &opts).map_err(|e| format!("{e:?}"))?;
        self.do_fetch(request).await
    }

    async fn do_fetch(&self, request: Request) -> Result<Response, String> {
        let window = crate::dom::window();
        let resp_val = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| format!("fetch error: {e:?}"))?;
        let resp: Response = resp_val.dyn_into().map_err(|_| "not a Response".to_string())?;

        if !resp.ok() {
            let status = resp.status();
            let text = self.text(resp).await.unwrap_or_default();
            return Err(format!("HTTP {status}: {text}"));
        }
        Ok(resp)
    }

    async fn text(&self, resp: Response) -> Result<String, String> {
        let text = JsFuture::from(resp.text().map_err(|e| format!("{e:?}"))?)
            .await
            .map_err(|e| format!("{e:?}"))?;
        text.as_string().ok_or_else(|| "not a string".to_string())
    }
}

/// Generate a simple browser fingerprint from available APIs.
pub fn browser_fingerprint() -> String {
    let window = crate::dom::window();
    let nav = window.navigator();
    let ua = nav.user_agent().unwrap_or_default();
    let lang = nav.language().unwrap_or_default();
    let cores = nav.hardware_concurrency() as u32;

    // Simple hash
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in format!("{ua}|{lang}|{cores}").bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
