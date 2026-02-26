use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

/// Compute "nice" tick values for an axis range.
fn nice_ticks(min: f32, max: f32, target_count: usize) -> (Vec<f32>, f32) {
    let range = max - min;
    if range.abs() < 1e-10 || target_count == 0 {
        return (vec![], 1.0);
    }
    let rough_step = range / target_count as f32;
    let mag = 10.0f32.powf(rough_step.log10().floor());
    let normalized = rough_step / mag;
    let nice_step = if normalized < 1.5 {
        mag
    } else if normalized < 3.5 {
        mag * 2.0
    } else if normalized < 7.5 {
        mag * 5.0
    } else {
        mag * 10.0
    };
    let start = (min / nice_step).ceil() * nice_step;
    let mut ticks = Vec::new();
    let mut v = start;
    while v <= max + nice_step * 0.01 {
        if v >= min - nice_step * 0.01 {
            ticks.push(v);
        }
        v += nice_step;
    }
    (ticks, nice_step)
}

/// Format a tick value with appropriate decimal places.
fn tick_label(v: f32, step: f32) -> String {
    if step >= 0.95 {
        format!("{:.0}", v)
    } else if step >= 0.095 {
        format!("{:.1}", v)
    } else if step >= 0.0095 {
        format!("{:.2}", v)
    } else {
        format!("{:.3}", v)
    }
}

/// Detect if the user prefers light color scheme.
fn is_light_mode() -> bool {
    web_sys::window()
        .and_then(|w| w.match_media("(prefers-color-scheme: light)").ok().flatten())
        .map(|m| m.matches())
        .unwrap_or(false)
}

/// A data point in the 3D parameter space.
#[derive(Clone, Debug)]
pub struct DataPoint {
    pub log_g2: f32,   // X: log10(g²)
    pub s_ee: f32,     // Y: entanglement entropy
    pub a_cut: f32,    // Z: cut area
    pub age: f32,      // for animation (0 = new, grows each frame)
    pub label: Option<String>, // floating annotation (set for fresh compute results)
}

/// Track data bounds for auto-scaling.
struct Bounds {
    x_min: f32, x_max: f32,
    y_min: f32, y_max: f32,
    z_min: f32, z_max: f32,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            x_min: f32::INFINITY, x_max: f32::NEG_INFINITY,
            y_min: f32::INFINITY, y_max: f32::NEG_INFINITY,
            z_min: f32::INFINITY, z_max: f32::NEG_INFINITY,
        }
    }
}

impl Bounds {
    fn update(&mut self, p: &DataPoint) {
        self.x_min = self.x_min.min(p.log_g2);
        self.x_max = self.x_max.max(p.log_g2);
        self.y_min = self.y_min.min(p.s_ee);
        self.y_max = self.y_max.max(p.s_ee);
        self.z_min = self.z_min.min(p.a_cut);
        self.z_max = self.z_max.max(p.a_cut);
    }

    fn normalize(&self, p: &DataPoint) -> [f32; 3] {
        [
            Self::map(p.log_g2, self.x_min, self.x_max),
            Self::map(p.s_ee, self.y_min, self.y_max),
            Self::map(p.a_cut, self.z_min, self.z_max),
        ]
    }

    fn map(v: f32, min: f32, max: f32) -> f32 {
        let range = max - min;
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((v - min) / range - 0.5) * 4.0
    }

    fn t_color(&self, a_cut: f32) -> f32 {
        let range = self.z_max - self.z_min;
        if range.abs() < 1e-10 {
            return 0.5;
        }
        ((a_cut - self.z_min) / range).clamp(0.0, 1.0)
    }
}

/// Camera state for orbit controls.
pub struct OrbitCamera {
    pub theta: f32,
    pub phi: f32,
    pub distance: f32,
    pub user_dragging: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            theta: 0.8,
            phi: 0.6,
            distance: 5.0,
            user_dragging: false,
        }
    }
}

impl OrbitCamera {
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.theta += dx * 0.01;
        self.phi = (self.phi - dy * 0.01).clamp(0.1, std::f32::consts::PI - 0.1);
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 + delta * 0.001)).clamp(2.0, 15.0);
    }

    fn project(&self, p: [f32; 3], w: f32, h: f32) -> (f32, f32, f32) {
        let (st, ct) = (self.theta.sin(), self.theta.cos());
        let (sp, cp) = (self.phi.sin(), self.phi.cos());

        let rx = p[0] * ct + p[2] * st;
        let rz = -p[0] * st + p[2] * ct;
        let ry_tmp = p[1];

        let ry = ry_tmp * cp - rz * sp;
        let rz2 = ry_tmp * sp + rz * cp;

        let z = rz2 + self.distance;
        let fov_scale = 2.0;
        if z < 0.1 {
            return (w / 2.0, h / 2.0, 0.0);
        }
        let scale = fov_scale * h / z;
        let sx = w / 2.0 + rx * scale;
        let sy = h / 2.0 - ry * scale;
        (sx, sy, z)
    }
}

/// Canvas2D-based 3D renderer for the parameter space.
pub struct Renderer {
    ctx: CanvasRenderingContext2d,
    canvas: HtmlCanvasElement,
    pub camera: OrbitCamera,
    pub points: Vec<DataPoint>,
    bounds: Bounds,
    /// CSS pixel dimensions (logical).
    width: u32,
    height: u32,
    dpr: f64,
    frame: u64,
    /// Mouse position in CSS pixels for hover tooltip.
    hover_pos: Option<(f32, f32)>,
}

fn device_pixel_ratio() -> f64 {
    web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
}

impl Renderer {
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let dpr = device_pixel_ratio();
        let width = canvas.client_width() as u32;
        let height = canvas.client_height() as u32;
        // Set backing store to physical pixels
        canvas.set_width((width as f64 * dpr) as u32);
        canvas.set_height((height as f64 * dpr) as u32);

        let ctx = canvas
            .get_context("2d")
            .map_err(|e| format!("{e:?}"))?
            .ok_or("no 2d context")?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| "not a 2d context")?;

        // Scale context so drawing ops use CSS pixels
        ctx.scale(dpr, dpr).ok();

        web_sys::console::log_1(
            &format!("canvas2d renderer: {}x{} @{dpr}x", width, height).into(),
        );

        Ok(Renderer {
            ctx,
            canvas,
            camera: OrbitCamera::default(),
            points: Vec::new(),
            bounds: Bounds::default(),
            width,
            height,
            dpr,
            frame: 0,
            hover_pos: None,
        })
    }

    pub fn add_point(&mut self, log_g2: f64, s_ee: f64, a_cut: f64) {
        self.add_point_labeled(log_g2, s_ee, a_cut, None);
    }

    pub fn add_point_labeled(&mut self, log_g2: f64, s_ee: f64, a_cut: f64, label: Option<String>) {
        let p = DataPoint {
            log_g2: log_g2 as f32,
            s_ee: s_ee as f32,
            a_cut: a_cut as f32,
            age: 0.0,
            label,
        };
        self.bounds.update(&p);
        self.points.push(p);
    }

    pub fn bounds_z_min(&self) -> f32 {
        self.bounds.z_min
    }

    pub fn bounds_z_range(&self) -> f32 {
        self.bounds.z_max - self.bounds.z_min
    }

    pub fn set_hover(&mut self, x: f32, y: f32) {
        self.hover_pos = Some((x, y));
    }

    pub fn clear_hover(&mut self) {
        self.hover_pos = None;
    }

    pub fn render(&mut self) {
        self.frame += 1;

        let cw = self.canvas.client_width() as u32;
        let ch = self.canvas.client_height() as u32;
        let dpr = device_pixel_ratio();
        if cw != self.width || ch != self.height || dpr != self.dpr {
            self.width = cw;
            self.height = ch;
            self.dpr = dpr;
            self.canvas.set_width((cw as f64 * dpr) as u32);
            self.canvas.set_height((ch as f64 * dpr) as u32);
            self.ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0).ok();
        }

        let w = self.width as f32;
        let h = self.height as f32;

        // Auto-rotate when not dragging
        if !self.camera.user_dragging {
            self.camera.theta += 0.002;
        }

        let light = is_light_mode();
        let bg = if light { "#f4f4f0" } else { "#0a0a10" };
        self.ctx.set_fill_style_str(bg);
        self.ctx.fill_rect(0.0, 0.0, w as f64, h as f64);

        // Draw grid + tick marks
        self.draw_grid(w, h, light);
        self.draw_ticks(w, h, light);

        // Age points
        for p in &mut self.points {
            p.age = (p.age + 0.016).min(3.0); // up to 3s for full ring fade
        }

        // Project all points (needed for connections + sorting + drawing)
        let projected: Vec<(usize, f32, f32, f32)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let pos = self.bounds.normalize(p);
                let (sx, sy, z) = self.camera.project(pos, w, h);
                (i, sx, sy, z)
            })
            .collect();

        // Sort back-to-front
        let mut draw_order: Vec<usize> = (0..projected.len()).collect();
        draw_order.sort_by(|&a, &b| {
            projected[b].3.partial_cmp(&projected[a].3).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Draw connection lines between nearby points (constellation effect)
        self.draw_connections(&projected, light);

        // Best-fit regression line (behind points)
        self.draw_regression_line(w, h, light);

        // Draw points
        for &idx in &draw_order {
            let (i, sx, sy, z) = projected[idx];
            let p = &self.points[i];

            if z < 0.1 {
                continue;
            }

            let t = self.bounds.t_color(p.a_cut);
            let (cr, cg, cb) = if light { retro_color_light(t) } else { retro_color(t) };

            // Depth cueing: farther points are dimmer
            let depth = (0.4 + 0.6 * (1.0 - (z - 3.0).max(0.0) / 6.0)).clamp(0.4, 1.0);
            let freshness = (1.0 - p.age).max(0.0);
            let alpha = (0.55 + 0.45 * freshness) * depth;
            let size = (2.5 + 3.5 * freshness) * (4.0 / z);

            // Expanding sonar ring on fresh points (age < 1.5s)
            if p.age < 1.5 {
                let ring_t = p.age / 1.5;
                let ring_radius = size * (1.5 + ring_t * 8.0);
                let ring_alpha = 0.3 * (1.0 - ring_t) * depth;
                self.ctx.set_stroke_style_str(
                    &format!("rgba({cr},{cg},{cb},{ring_alpha:.2})"),
                );
                self.ctx.set_line_width(1.0);
                self.ctx.begin_path();
                self.ctx.arc(sx as f64, sy as f64, ring_radius as f64, 0.0, std::f64::consts::TAU).ok();
                self.ctx.stroke();
            }

            // Second ring (delayed, offset)
            if p.age > 0.3 && p.age < 2.0 {
                let ring_t = (p.age - 0.3) / 1.7;
                let ring_radius = size * (1.0 + ring_t * 6.0);
                let ring_alpha = 0.15 * (1.0 - ring_t) * depth;
                self.ctx.set_stroke_style_str(
                    &format!("rgba({cr},{cg},{cb},{ring_alpha:.2})"),
                );
                self.ctx.set_line_width(0.5);
                self.ctx.begin_path();
                self.ctx.arc(sx as f64, sy as f64, ring_radius as f64, 0.0, std::f64::consts::TAU).ok();
                self.ctx.stroke();
            }

            // Solid dot
            self.ctx.set_fill_style_str(
                &format!("rgba({cr},{cg},{cb},{alpha:.2})"),
            );
            self.ctx.begin_path();
            self.ctx.arc(sx as f64, sy as f64, size as f64, 0.0, std::f64::consts::TAU).ok();
            self.ctx.fill();

            // Crosshair on very fresh points (< 0.5s)
            if p.age < 0.5 {
                let ch_alpha = (1.0 - p.age / 0.5) * 0.6 * depth;
                let ch_size = size * 4.0;
                self.ctx.set_stroke_style_str(
                    &format!("rgba({cr},{cg},{cb},{ch_alpha:.2})"),
                );
                self.ctx.set_line_width(1.0);
                self.ctx.begin_path();
                self.ctx.move_to((sx - ch_size) as f64, sy as f64);
                self.ctx.line_to((sx + ch_size) as f64, sy as f64);
                self.ctx.stroke();
                self.ctx.begin_path();
                self.ctx.move_to(sx as f64, (sy - ch_size) as f64);
                self.ctx.line_to(sx as f64, (sy + ch_size) as f64);
                self.ctx.stroke();
            }
        }

        // Floating annotations on recent points (drawn on top of everything)
        self.draw_annotations(&projected, &draw_order, light);

        // Axis labels (with subtitles)
        self.draw_labels(w, h, light);

        // Hover tooltip (on top of everything)
        self.draw_hover_tooltip(&projected, w, h, light);

        // Color legend (screen-space overlay)
        self.draw_color_legend(w, h, light);

        // Stats HUD (screen-space overlay)
        self.draw_stats_hud(w, h, light);

        // Explainer (top-left)
        self.draw_explainer(w, h, light);
    }

    /// Draw floating text annotations near fresh labeled points.
    fn draw_annotations(
        &self,
        projected: &[(usize, f32, f32, f32)],
        draw_order: &[usize],
        light: bool,
    ) {
        let label_color = if light {
            "rgba(50,50,46,"
        } else {
            "rgba(200,200,210,"
        };
        let leader_color = if light {
            "rgba(80,80,76,"
        } else {
            "rgba(100,100,120,"
        };

        self.ctx.set_font("10px 'IBM Plex Mono', monospace");
        self.ctx.set_text_baseline("middle");

        // Draw front-to-back so nearer labels overlay farther ones
        for &idx in draw_order.iter().rev() {
            let (i, sx, sy, z) = projected[idx];
            let p = &self.points[i];

            // Only draw for labeled points that are still fresh enough
            let Some(ref label) = p.label else { continue };
            if p.age > 3.0 || z < 0.1 {
                continue;
            }

            let fade = 1.0 - (p.age / 3.0);
            let fade = fade * fade; // ease-out
            if fade < 0.01 { continue; }

            // Offset: label floats up and to the right, drifts upward with age
            let drift_y = -p.age * 4.0; // drift up over time
            let offset_x: f32 = 14.0;
            let offset_y: f32 = -12.0 + drift_y;
            let lx = sx + offset_x;
            let ly = sy + offset_y;

            // Leader line from point to label
            self.ctx.set_stroke_style_str(
                &format!("{leader_color}{:.2})", fade * 0.4),
            );
            self.ctx.set_line_width(0.5);
            self.ctx.begin_path();
            self.ctx.move_to(sx as f64, sy as f64);
            self.ctx.line_to(lx as f64, ly as f64);
            self.ctx.stroke();

            // Split label into lines (use | as separator)
            let lines: Vec<&str> = label.split('|').collect();
            for (li, line) in lines.iter().enumerate() {
                let line_y = ly + li as f32 * 12.0;

                // Text
                self.ctx.set_fill_style_str(
                    &format!("{label_color}{:.2})", fade * 0.8),
                );
                self.ctx.fill_text(line, lx as f64, line_y as f64).ok();
            }
        }

        self.ctx.set_text_baseline("alphabetic");
    }

    /// Draw faint lines between nearby projected points.
    fn draw_connections(&self, projected: &[(usize, f32, f32, f32)], light: bool) {
        if projected.len() < 2 || projected.len() > 2000 {
            return;
        }

        let threshold_sq = 2500.0f32; // ~50px distance
        let base_color = if light { (26, 138, 68) } else { (68, 204, 102) };

        // Only check a subset of pairs for performance (sample every Nth point)
        let step = (projected.len() / 200).max(1);

        for i in (0..projected.len()).step_by(step) {
            let (_, x1, y1, z1) = projected[i];
            if z1 < 0.1 { continue; }

            for j in (i + 1..projected.len()).step_by(step) {
                let (_, x2, y2, z2) = projected[j];
                if z2 < 0.1 { continue; }

                let dx = x2 - x1;
                let dy = y2 - y1;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < threshold_sq && dist_sq > 100.0 {
                    let proximity = 1.0 - (dist_sq / threshold_sq).sqrt();
                    let depth = (0.3 + 0.7 * (1.0 - ((z1 + z2) * 0.5 - 3.0).max(0.0) / 6.0)).clamp(0.2, 1.0);
                    let alpha = proximity * 0.15 * depth;

                    let (cr, cg, cb) = base_color;
                    self.ctx.set_stroke_style_str(
                        &format!("rgba({cr},{cg},{cb},{alpha:.3})"),
                    );
                    self.ctx.set_line_width(0.5);
                    self.ctx.begin_path();
                    self.ctx.move_to(x1 as f64, y1 as f64);
                    self.ctx.line_to(x2 as f64, y2 as f64);
                    self.ctx.stroke();
                }
            }
        }
    }

    fn draw_grid(&self, w: f32, h: f32, light: bool) {
        let r = 2.0f32;

        let axes = if light {
            [
                ([-r, 0.0, 0.0], [r, 0.0, 0.0], "rgba(180,70,70,0.6)"),
                ([0.0, -r, 0.0], [0.0, r, 0.0], "rgba(50,150,50,0.6)"),
                ([0.0, 0.0, -r], [0.0, 0.0, r], "rgba(70,70,180,0.6)"),
            ]
        } else {
            [
                ([-r, 0.0, 0.0], [r, 0.0, 0.0], "rgba(200,110,110,0.7)"),
                ([0.0, -r, 0.0], [0.0, r, 0.0], "rgba(110,200,110,0.7)"),
                ([0.0, 0.0, -r], [0.0, 0.0, r], "rgba(110,110,200,0.7)"),
            ]
        };

        for (from, to, color) in &axes {
            let (x1, y1, _) = self.camera.project(*from, w, h);
            let (x2, y2, _) = self.camera.project(*to, w, h);
            self.ctx.set_stroke_style_str(color);
            self.ctx.set_line_width(1.5);
            self.ctx.begin_path();
            self.ctx.move_to(x1 as f64, y1 as f64);
            self.ctx.line_to(x2 as f64, y2 as f64);
            self.ctx.stroke();
        }

        let grid_color = if light { "rgba(160,160,150,0.5)" } else { "rgba(55,55,75,0.7)" };
        self.ctx.set_stroke_style_str(grid_color);
        self.ctx.set_line_width(0.5);
        let n = 8;
        for i in 0..=n {
            let f = -r + (4.0 * i as f32 / n as f32);
            let (x1, y1, _) = self.camera.project([f, 0.0, -r], w, h);
            let (x2, y2, _) = self.camera.project([f, 0.0, r], w, h);
            self.ctx.begin_path();
            self.ctx.move_to(x1 as f64, y1 as f64);
            self.ctx.line_to(x2 as f64, y2 as f64);
            self.ctx.stroke();

            let (x1, y1, _) = self.camera.project([-r, 0.0, f], w, h);
            let (x2, y2, _) = self.camera.project([r, 0.0, f], w, h);
            self.ctx.begin_path();
            self.ctx.move_to(x1 as f64, y1 as f64);
            self.ctx.line_to(x2 as f64, y2 as f64);
            self.ctx.stroke();
        }
    }

    /// Draw numeric tick marks along each axis.
    fn draw_ticks(&self, w: f32, h: f32, light: bool) {
        if self.points.is_empty() {
            return;
        }

        let tick_color = if light {
            "rgba(120,120,115,0.6)"
        } else {
            "rgba(90,90,110,0.5)"
        };
        self.ctx.set_font("8px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(tick_color);
        self.ctx.set_stroke_style_str(tick_color);
        self.ctx.set_line_width(0.5);

        let tick_len = 0.12f32;

        // X axis ticks (log_g2)
        let (x_ticks, x_step) = nice_ticks(self.bounds.x_min, self.bounds.x_max, 4);
        self.ctx.set_text_align("center");
        self.ctx.set_text_baseline("top");
        for &v in &x_ticks {
            let n = Bounds::map(v, self.bounds.x_min, self.bounds.x_max);
            let (px, py, _) = self.camera.project([n, 0.0, 0.0], w, h);
            let (tx, ty, _) = self.camera.project([n, -tick_len, 0.0], w, h);
            self.ctx.begin_path();
            self.ctx.move_to(px as f64, py as f64);
            self.ctx.line_to(tx as f64, ty as f64);
            self.ctx.stroke();
            self.ctx
                .fill_text(&tick_label(v, x_step), tx as f64, (ty + 2.0) as f64)
                .ok();
        }

        // Y axis ticks (S_EE)
        let (y_ticks, y_step) = nice_ticks(self.bounds.y_min, self.bounds.y_max, 4);
        self.ctx.set_text_align("right");
        self.ctx.set_text_baseline("middle");
        for &v in &y_ticks {
            let n = Bounds::map(v, self.bounds.y_min, self.bounds.y_max);
            let (px, py, _) = self.camera.project([0.0, n, 0.0], w, h);
            let (tx, ty, _) = self.camera.project([-tick_len, n, 0.0], w, h);
            self.ctx.begin_path();
            self.ctx.move_to(px as f64, py as f64);
            self.ctx.line_to(tx as f64, ty as f64);
            self.ctx.stroke();
            self.ctx
                .fill_text(&tick_label(v, y_step), (tx - 3.0) as f64, ty as f64)
                .ok();
        }

        // Z axis ticks (A_cut)
        let (z_ticks, z_step) = nice_ticks(self.bounds.z_min, self.bounds.z_max, 4);
        self.ctx.set_text_align("right");
        self.ctx.set_text_baseline("middle");
        for &v in &z_ticks {
            let n = Bounds::map(v, self.bounds.z_min, self.bounds.z_max);
            let (px, py, _) = self.camera.project([0.0, 0.0, n], w, h);
            let (tx, ty, _) = self.camera.project([-tick_len, 0.0, n], w, h);
            self.ctx.begin_path();
            self.ctx.move_to(px as f64, py as f64);
            self.ctx.line_to(tx as f64, ty as f64);
            self.ctx.stroke();
            self.ctx
                .fill_text(&tick_label(v, z_step), (tx - 3.0) as f64, ty as f64)
                .ok();
        }

        self.ctx.set_text_align("left");
        self.ctx.set_text_baseline("alphabetic");
    }

    /// Draw the S_EE = slope * A_cut best-fit regression line in 3D.
    fn draw_regression_line(&self, w: f32, h: f32, light: bool) {
        if self.points.len() < 10 {
            return;
        }

        let (slope, intercept, _r2) = self.compute_regression();
        if slope.abs() < 1e-15 {
            return;
        }

        // Draw line across the A_cut range at the mean log_g2
        let mean_x: f32 =
            self.points.iter().map(|p| p.log_g2).sum::<f32>() / self.points.len() as f32;
        let norm_x = Bounds::map(mean_x, self.bounds.x_min, self.bounds.x_max);

        let segments = 24;
        let color = if light {
            "rgba(180,150,40,0.5)"
        } else {
            "rgba(204,170,51,0.4)"
        };
        self.ctx.set_stroke_style_str(color);
        self.ctx.set_line_width(1.5);

        // Dashed line
        let dash = js_sys::Array::new();
        dash.push(&6.0.into());
        dash.push(&4.0.into());
        self.ctx.set_line_dash(&dash).ok();

        self.ctx.begin_path();
        let mut started = false;
        for i in 0..=segments {
            let t = i as f32 / segments as f32;
            let a_cut = self.bounds.z_min + t * (self.bounds.z_max - self.bounds.z_min);
            let s_ee = (slope * a_cut as f64 + intercept) as f32;

            if s_ee < self.bounds.y_min || s_ee > self.bounds.y_max {
                continue;
            }

            let norm_y = Bounds::map(s_ee, self.bounds.y_min, self.bounds.y_max);
            let norm_z = Bounds::map(a_cut, self.bounds.z_min, self.bounds.z_max);
            let (sx, sy, z) = self.camera.project([norm_x, norm_y, norm_z], w, h);
            if z < 0.1 {
                continue;
            }

            if !started {
                self.ctx.move_to(sx as f64, sy as f64);
                started = true;
            } else {
                self.ctx.line_to(sx as f64, sy as f64);
            }
        }
        self.ctx.stroke();
        self.ctx.set_line_dash(&js_sys::Array::new()).ok();

        // Draw equation label near the midpoint
        let mid_a = (self.bounds.z_min + self.bounds.z_max) / 2.0;
        let mid_s = (slope * mid_a as f64 + intercept) as f32;
        if mid_s >= self.bounds.y_min && mid_s <= self.bounds.y_max {
            let norm_y = Bounds::map(mid_s, self.bounds.y_min, self.bounds.y_max);
            let norm_z = Bounds::map(mid_a, self.bounds.z_min, self.bounds.z_max);
            let (sx, sy, z) = self.camera.project([norm_x, norm_y, norm_z], w, h);
            if z > 0.1 {
                let g_n = 1.0 / (4.0 * slope);
                let label = format!("S = A / {:.0}", 4.0 * g_n);
                let label_color = if light {
                    "rgba(140,120,30,0.7)"
                } else {
                    "rgba(204,170,51,0.7)"
                };
                let sub_color = if light {
                    "rgba(140,120,30,0.45)"
                } else {
                    "rgba(204,170,51,0.35)"
                };
                self.ctx.set_font("10px 'IBM Plex Mono', monospace");
                self.ctx.set_fill_style_str(label_color);
                self.ctx
                    .fill_text(&label, (sx + 10.0) as f64, (sy - 6.0) as f64)
                    .ok();
                // Subtitle: "best fit" hint
                self.ctx.set_font("8px 'IBM Plex Mono', monospace");
                self.ctx.set_fill_style_str(sub_color);
                self.ctx
                    .fill_text(
                        &format!("G_N \u{2248} {:.2}", g_n),
                        (sx + 10.0) as f64,
                        (sy + 7.0) as f64,
                    )
                    .ok();
            }
        }
    }

    /// Linear regression: S_EE = slope * A_cut + intercept. Returns (slope, intercept, R²).
    fn compute_regression(&self) -> (f64, f64, f64) {
        if self.points.len() < 2 {
            return (0.0, 0.0, 0.0);
        }
        let n = self.points.len() as f64;
        let sum_x: f64 = self.points.iter().map(|p| p.a_cut as f64).sum();
        let sum_y: f64 = self.points.iter().map(|p| p.s_ee as f64).sum();
        let sum_xy: f64 = self
            .points
            .iter()
            .map(|p| p.a_cut as f64 * p.s_ee as f64)
            .sum();
        let sum_xx: f64 = self.points.iter().map(|p| (p.a_cut as f64).powi(2)).sum();
        let sum_yy: f64 = self.points.iter().map(|p| (p.s_ee as f64).powi(2)).sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-20 {
            return (0.0, 0.0, 0.0);
        }
        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        let num_r = n * sum_xy - sum_x * sum_y;
        let den_r = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
        let r2 = if den_r.abs() < 1e-20 {
            0.0
        } else {
            (num_r / den_r).powi(2)
        };

        (slope, intercept, r2)
    }

    /// Draw a vertical color legend showing the A_cut color mapping.
    fn draw_color_legend(&self, w: f32, h: f32, light: bool) {
        if self.points.is_empty() {
            return;
        }

        let bar_x = w - 34.0;
        let bar_top = h * 0.35;
        let bar_height = 80.0;
        let bar_width = 6.0;
        let segments = 20;

        // Gradient bar
        for i in 0..segments {
            let t0 = i as f32 / segments as f32;
            let t1 = (i + 1) as f32 / segments as f32;
            let (r, g, b) = if light {
                retro_color_light(1.0 - t0)
            } else {
                retro_color(1.0 - t0)
            };
            self.ctx
                .set_fill_style_str(&format!("rgba({r},{g},{b},0.7)"));
            let y0 = bar_top + t0 * bar_height;
            let seg_h = (t1 - t0) * bar_height;
            self.ctx.fill_rect(
                bar_x as f64,
                y0 as f64,
                bar_width as f64,
                seg_h as f64 + 0.5,
            );
        }

        let label_color = if light {
            "rgba(100,100,95,0.7)"
        } else {
            "rgba(140,140,160,0.6)"
        };
        self.ctx.set_font("8px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(label_color);
        self.ctx.set_text_align("right");
        self.ctx.set_text_baseline("middle");

        self.ctx
            .fill_text(
                &format!("{:.1}", self.bounds.z_max),
                (bar_x - 3.0) as f64,
                bar_top as f64,
            )
            .ok();
        self.ctx
            .fill_text(
                &format!("{:.1}", self.bounds.z_min),
                (bar_x - 3.0) as f64,
                (bar_top + bar_height) as f64,
            )
            .ok();

        // Title above bar
        self.ctx.set_text_align("center");
        self.ctx.set_text_baseline("bottom");
        self.ctx
            .fill_text(
                "A\u{2202}",
                (bar_x + bar_width / 2.0) as f64,
                (bar_top - 4.0) as f64,
            )
            .ok();

        self.ctx.set_text_align("left");
        self.ctx.set_text_baseline("alphabetic");
    }

    /// Draw tooltip for the nearest point under the mouse cursor.
    fn draw_hover_tooltip(
        &self,
        projected: &[(usize, f32, f32, f32)],
        w: f32,
        h: f32,
        light: bool,
    ) {
        let Some((mx, my)) = self.hover_pos else {
            return;
        };
        if projected.is_empty() {
            return;
        }

        // Find nearest projected point
        let mut best_idx = 0;
        let mut best_dist_sq = f32::INFINITY;
        for (i, &(_, sx, sy, z)) in projected.iter().enumerate() {
            if z < 0.1 {
                continue;
            }
            let dx = sx - mx;
            let dy = sy - my;
            let d = dx * dx + dy * dy;
            if d < best_dist_sq {
                best_dist_sq = d;
                best_idx = i;
            }
        }

        // Only show if within ~30px
        if best_dist_sq > 900.0 {
            return;
        }

        let (pi, sx, sy, _z) = projected[best_idx];
        let p = &self.points[pi];

        // Highlight ring
        let t = self.bounds.t_color(p.a_cut);
        let (cr, cg, cb) = if light {
            retro_color_light(t)
        } else {
            retro_color(t)
        };
        self.ctx
            .set_stroke_style_str(&format!("rgba({cr},{cg},{cb},0.8)"));
        self.ctx.set_line_width(1.5);
        self.ctx.begin_path();
        self.ctx
            .arc(sx as f64, sy as f64, 8.0, 0.0, std::f64::consts::TAU)
            .ok();
        self.ctx.stroke();

        // Tooltip content
        let lines = [
            format!("log(g\u{00B2}) {:.2}", p.log_g2),
            format!("S_EE    {:.4}", p.s_ee),
            format!("A_cut   {:.2}", p.a_cut),
        ];

        let line_h = 13.0f32;
        let pad = 5.0f32;
        let box_w = 120.0f32;
        let box_h = lines.len() as f32 * line_h + pad * 2.0;

        // Position tooltip, keeping it on screen
        let mut tx = sx + 14.0;
        let mut ty = sy - box_h / 2.0;
        if tx + box_w > w - 8.0 {
            tx = sx - 14.0 - box_w;
        }
        if ty < 40.0 {
            ty = 40.0;
        }
        if ty + box_h > h - 60.0 {
            ty = h - 60.0 - box_h;
        }

        // Background
        let bg_color = if light {
            "rgba(234,234,230,0.92)"
        } else {
            "rgba(16,16,24,0.92)"
        };
        let border_color = if light {
            "rgba(200,200,192,0.8)"
        } else {
            "rgba(60,60,80,0.6)"
        };
        self.ctx.set_fill_style_str(bg_color);
        self.ctx
            .fill_rect(tx as f64, ty as f64, box_w as f64, box_h as f64);
        self.ctx.set_stroke_style_str(border_color);
        self.ctx.set_line_width(0.5);
        self.ctx
            .stroke_rect(tx as f64, ty as f64, box_w as f64, box_h as f64);

        // Leader line from point to tooltip
        let leader = if light {
            "rgba(120,120,115,0.4)"
        } else {
            "rgba(90,90,110,0.3)"
        };
        self.ctx.set_stroke_style_str(leader);
        self.ctx.set_line_width(0.5);
        self.ctx.begin_path();
        self.ctx.move_to(sx as f64, sy as f64);
        self.ctx
            .line_to(tx as f64, (ty + box_h / 2.0) as f64);
        self.ctx.stroke();

        // Text
        let text_color = if light {
            "rgba(40,40,36,0.9)"
        } else {
            "rgba(200,200,220,0.9)"
        };
        self.ctx.set_font("9px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(text_color);
        self.ctx.set_text_baseline("top");
        for (i, line) in lines.iter().enumerate() {
            let ly = ty + pad + i as f32 * line_h;
            self.ctx
                .fill_text(line, (tx + pad) as f64, ly as f64)
                .ok();
        }
        self.ctx.set_text_baseline("alphabetic");
    }

    fn draw_labels(&self, w: f32, h: f32, light: bool) {
        let (lx, ly, lz) = if light {
            ("rgba(180,70,70,0.8)", "rgba(50,150,50,0.8)", "rgba(70,70,180,0.8)")
        } else {
            ("rgba(210,130,130,0.8)", "rgba(130,210,130,0.8)", "rgba(130,130,210,0.8)")
        };

        let sub_color = if light {
            "rgba(120,120,115,0.5)"
        } else {
            "rgba(120,120,140,0.4)"
        };

        // X axis: log(g²)
        self.ctx.set_font("11px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(lx);
        let (x, y, _) = self.camera.project([2.3, 0.0, 0.0], w, h);
        self.ctx.fill_text("log(g\u{00B2})", x as f64, y as f64).ok();
        self.ctx.set_font("8px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(sub_color);
        self.ctx
            .fill_text("coupling", x as f64, (y + 12.0) as f64)
            .ok();

        // Y axis: S_EE
        self.ctx.set_font("11px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(ly);
        let (x, y, _) = self.camera.project([0.0, 2.3, 0.0], w, h);
        self.ctx.fill_text("S_EE", x as f64, y as f64).ok();
        self.ctx.set_font("8px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(sub_color);
        self.ctx
            .fill_text("entanglement entropy", x as f64, (y + 12.0) as f64)
            .ok();

        // Z axis: A_∂
        self.ctx.set_font("11px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(lz);
        let (x, y, _) = self.camera.project([0.0, 0.0, 2.3], w, h);
        self.ctx.fill_text("A_\u{2202}", x as f64, y as f64).ok();
        self.ctx.set_font("8px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(sub_color);
        self.ctx
            .fill_text("cut area", x as f64, (y + 12.0) as f64)
            .ok();
    }

    /// Draw a HUD panel with regression stats and physics interpretation.
    fn draw_stats_hud(&self, w: f32, h: f32, light: bool) {
        if self.points.len() < 10 {
            return;
        }

        let (slope, intercept, r2) = self.compute_regression();
        if slope.abs() < 1e-15 {
            return;
        }

        let g_n = 1.0 / (4.0 * slope);
        let n = self.points.len();

        // Build lines
        let lines = [
            ("Ryu-Takayanagi fit", true),
            (&format!("S_EE = A / {:.1}", 4.0 * g_n), false),
            ("", false),
            (&format!("slope     {:.6}", slope), false),
            (&format!("intercept {:.4}", intercept), false),
            (&format!("R\u{00B2}        {:.4}", r2), false),
            (&format!("G_N       {:.2}", g_n), false),
            (&format!("n         {}", n), false),
        ];

        let line_h = 13.0f32;
        let pad = 8.0f32;
        let box_w = 160.0f32;
        let box_h = lines.len() as f32 * line_h + pad * 2.0;
        let bx = 12.0f32;
        let by = h - box_h - 12.0;

        // Background
        let bg = if light {
            "rgba(244,244,240,0.75)"
        } else {
            "rgba(10,10,18,0.75)"
        };
        let border = if light {
            "rgba(200,200,192,0.4)"
        } else {
            "rgba(50,50,70,0.4)"
        };
        self.ctx.set_fill_style_str(bg);
        self.ctx
            .fill_rect(bx as f64, by as f64, box_w as f64, box_h as f64);
        self.ctx.set_stroke_style_str(border);
        self.ctx.set_line_width(0.5);
        self.ctx
            .stroke_rect(bx as f64, by as f64, box_w as f64, box_h as f64);

        // R² quality indicator — colored bar along top of box
        let bar_w = (box_w - 2.0) * r2.clamp(0.0, 1.0) as f32;
        let bar_color = if r2 > 0.9 {
            if light { "rgba(40,140,60,0.5)" } else { "rgba(80,220,100,0.4)" }
        } else if r2 > 0.7 {
            if light { "rgba(160,140,30,0.5)" } else { "rgba(204,170,51,0.4)" }
        } else {
            if light { "rgba(180,70,70,0.5)" } else { "rgba(220,90,90,0.4)" }
        };
        self.ctx.set_fill_style_str(bar_color);
        self.ctx
            .fill_rect((bx + 1.0) as f64, by as f64, bar_w as f64, 2.0);

        let text_color = if light {
            "rgba(50,50,46,0.8)"
        } else {
            "rgba(180,180,200,0.7)"
        };
        let header_color = if light {
            "rgba(50,50,46,0.9)"
        } else {
            "rgba(200,200,220,0.85)"
        };
        let dim_color = if light {
            "rgba(120,120,115,0.5)"
        } else {
            "rgba(100,100,120,0.45)"
        };

        self.ctx.set_text_baseline("top");
        self.ctx.set_text_align("left");

        for (i, (line, is_header)) in lines.iter().enumerate() {
            let ly = by + pad + i as f32 * line_h;
            if line.is_empty() {
                // Separator line
                self.ctx.set_stroke_style_str(dim_color);
                self.ctx.set_line_width(0.5);
                self.ctx.begin_path();
                self.ctx
                    .move_to((bx + pad) as f64, (ly + line_h * 0.5) as f64);
                self.ctx
                    .line_to((bx + box_w - pad) as f64, (ly + line_h * 0.5) as f64);
                self.ctx.stroke();
                continue;
            }
            if *is_header {
                self.ctx.set_font("9px 'IBM Plex Mono', monospace");
                self.ctx.set_fill_style_str(header_color);
            } else {
                self.ctx.set_font("8px 'IBM Plex Mono', monospace");
                self.ctx.set_fill_style_str(text_color);
            }
            self.ctx
                .fill_text(line, (bx + pad) as f64, ly as f64)
                .ok();
        }

        self.ctx.set_text_baseline("alphabetic");
    }

    /// Draw typographic explainer in the top-left corner.
    fn draw_explainer(&self, _w: f32, _h: f32, light: bool) {
        let x = 20.0f64;
        let mut y = 56.0f64;

        // Title — large, high contrast
        let title_color = if light {
            "rgba(30,30,28,0.85)"
        } else {
            "rgba(220,220,230,0.8)"
        };
        self.ctx.set_font("13px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(title_color);
        self.ctx.set_text_align("left");
        self.ctx.set_text_baseline("top");
        self.ctx.fill_text("Does gravity emerge from entanglement?", x, y).ok();

        // Thin rule
        y += 20.0;
        let rule_color = if light {
            "rgba(100,100,95,0.25)"
        } else {
            "rgba(120,120,140,0.2)"
        };
        self.ctx.set_stroke_style_str(rule_color);
        self.ctx.set_line_width(0.5);
        self.ctx.begin_path();
        self.ctx.move_to(x, y);
        self.ctx.line_to(x + 280.0, y);
        self.ctx.stroke();

        // Body lines — smaller, dimmer, generous spacing
        let body_color = if light {
            "rgba(70,70,65,0.6)"
        } else {
            "rgba(160,160,175,0.5)"
        };
        let em_color = if light {
            "rgba(50,50,46,0.75)"
        } else {
            "rgba(190,190,205,0.65)"
        };

        y += 8.0;
        let line_h = 15.0f64;

        // Line 1
        self.ctx.set_font("9px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(body_color);
        self.ctx.fill_text(
            "SU(2) lattice gauge theory on a triangulated 4-sphere.",
            x, y,
        ).ok();

        // Line 2
        y += line_h;
        self.ctx.set_fill_style_str(body_color);
        self.ctx.fill_text(
            "Each point: one partition of the ground state.",
            x, y,
        ).ok();

        // Line 3 — the key claim, slightly brighter
        y += line_h + 4.0;
        self.ctx.set_fill_style_str(em_color);
        self.ctx.fill_text(
            "If the Ryu-Takayanagi conjecture holds:",
            x, y,
        ).ok();

        // The formula — larger, prominent
        y += line_h + 2.0;
        let formula_color = if light {
            "rgba(140,120,30,0.8)"
        } else {
            "rgba(204,170,51,0.75)"
        };
        self.ctx.set_font("12px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(formula_color);
        self.ctx.fill_text(
            "S_EE  =  A_cut / 4G_N",
            x + 8.0, y,
        ).ok();

        // Closing line
        y += line_h + 6.0;
        self.ctx.set_font("9px 'IBM Plex Mono', monospace");
        self.ctx.set_fill_style_str(body_color);
        self.ctx.fill_text(
            "Points should collapse onto a tilted plane.",
            x, y,
        ).ok();

        y += line_h;
        self.ctx.set_fill_style_str(body_color);
        self.ctx.fill_text(
            "The slope gives Newton\u{2019}s constant.",
            x, y,
        ).ok();

        self.ctx.set_text_baseline("alphabetic");
    }
}

/// Retro terminal color ramp (dark mode): dark green → green → amber.
pub fn retro_color(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let s = t / 0.5;
        let r = (34.0 + s * 34.0) as u8;
        let g = (119.0 + s * 85.0) as u8;
        let b = (68.0 + s * 34.0) as u8;
        (r, g, b)
    } else {
        let s = (t - 0.5) / 0.5;
        let r = (68.0 + s * 136.0) as u8;
        let g = (204.0 - s * 34.0) as u8;
        let b = (102.0 - s * 51.0) as u8;
        (r, g, b)
    }
}

/// Color ramp for light mode: deeper green → teal → dark amber.
fn retro_color_light(t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let s = t / 0.5;
        let r = (20.0 + s * 20.0) as u8;
        let g = (100.0 + s * 40.0) as u8;
        let b = (60.0 + s * 20.0) as u8;
        (r, g, b)
    } else {
        let s = (t - 0.5) / 0.5;
        let r = (40.0 + s * 98.0) as u8;
        let g = (140.0 - s * 20.0) as u8;
        let b = (80.0 - s * 40.0) as u8;
        (r, g, b)
    }
}
