use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

/// Renders a defrag-style grid inside an inline bar.
/// Adapts to whatever CSS size its canvas is given.
pub struct DefragRenderer {
    ctx: CanvasRenderingContext2d,
    canvas: HtmlCanvasElement,
    /// Stable random permutation for scattering completed blocks.
    perm: Vec<u32>,
    last_n_cells: usize,
}

impl DefragRenderer {
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let ctx = canvas
            .get_context("2d")
            .map_err(|e| format!("{e:?}"))?
            .ok_or("no 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| "not CanvasRenderingContext2d")?;

        Ok(Self {
            ctx,
            canvas,
            perm: Vec::new(),
            last_n_cells: 0,
        })
    }

    /// Render the defrag grid.
    /// - `total`: total work units
    /// - `submitted`: total result rows submitted across all contributors
    /// - `consensus`: work units with verified consensus (status='complete')
    /// - `partial`: work units with results but no consensus yet
    pub fn render(&mut self, total: usize, submitted: usize, consensus: usize, partial: usize) {
        if total == 0 {
            return;
        }

        // Read CSS display size and derive canvas intrinsic size.
        // Each intrinsic pixel = ~2 CSS pixels → visible block.
        let display_w = self.canvas.client_width() as u32;
        let display_h = self.canvas.client_height() as u32;
        if display_w == 0 || display_h == 0 {
            return;
        }

        let cols = (display_w / 2).max(1);
        let rows = (display_h / 2).max(1);
        let n_cells = (cols * rows) as usize;

        if self.canvas.width() != cols || self.canvas.height() != rows {
            self.canvas.set_width(cols);
            self.canvas.set_height(rows);
        }

        if n_cells != self.last_n_cells {
            self.rebuild_perm(n_cells);
            self.last_n_cells = n_cells;
        }

        // Map proportions to cell counts.
        // Green = verified consensus, amber = has results but unverified,
        // blue = submitted (coverage estimate when DB hasn't tallied yet),
        // dark = no data.
        let complete_cells = (consensus as u64 * n_cells as u64 / total as u64) as usize;
        let partial_cells = (partial as u64 * n_cells as u64 / total as u64) as usize;

        // When DB status fields aren't populated yet, estimate coverage
        // from submitted results to show that work IS happening.
        let coverage_cells = if consensus == 0 && partial == 0 && submitted > 0 {
            // ~submitted/total of units have at least one result
            let est_touched = total.min(submitted);
            (est_touched as u64 * n_cells as u64 / total as u64) as usize
        } else {
            0
        };

        // Block colors (RGB)
        let green: [u8; 3] = [0x33, 0xbb, 0x55]; // verified consensus
        let amber: [u8; 3] = [0xbb, 0x99, 0x33]; // has results, unverified
        let blue: [u8; 3] = [0x33, 0x66, 0x99]; // submitted but unverified
        let dark: [u8; 3] = [0x2a, 0x2a, 0x38]; // no data — visible against page bg

        let mut buf = vec![0u8; n_cells * 4];
        for i in 0..n_cells {
            let offset = i * 4;
            let logical = self.perm[i] as usize;
            let [r, g, b] = if logical < complete_cells {
                green
            } else if logical < complete_cells + partial_cells {
                amber
            } else if logical < complete_cells + partial_cells + coverage_cells {
                blue
            } else {
                dark
            };
            buf[offset] = r;
            buf[offset + 1] = g;
            buf[offset + 2] = b;
            buf[offset + 3] = 255;
        }

        if let Ok(data) =
            ImageData::new_with_u8_clamped_array_and_sh(wasm_bindgen::Clamped(&buf), cols, rows)
        {
            self.ctx.put_image_data(&data, 0.0, 0.0).ok();
        }

        // Tooltip with full breakdown
        let pending = total.saturating_sub(consensus + partial);
        let coverage = submitted as f64 / total as f64;
        let title = if consensus > 0 || partial > 0 {
            format!(
                "{} work units: {} complete · {} in progress · {} pending\n\
                 {} results submitted ({:.1}× coverage)",
                fmt_num(total),
                fmt_num(consensus),
                fmt_num(partial),
                fmt_num(pending),
                fmt_num(submitted),
                coverage,
            )
        } else {
            format!(
                "{} work units — {} results submitted ({:.1}× coverage)\n\
                 awaiting consensus verification",
                fmt_num(total),
                fmt_num(submitted),
                coverage,
            )
        };
        self.canvas.set_title(&title);
    }

    fn rebuild_perm(&mut self, n: usize) {
        self.perm = (0..n as u32).collect();
        let mut rng: u64 = 0xcafe_babe_dead_beef;
        for i in (1..n).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            self.perm.swap(i, j);
        }
    }
}

/// Format a number with comma separators: 78875 → "78,875"
fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}
