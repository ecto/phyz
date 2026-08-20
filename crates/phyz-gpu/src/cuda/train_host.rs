//! The training kernels, compiled as host C++ and walked serially.
//!
//! `build.rs` compiles `cuda/phyz_train.cu` with the system C++ compiler and
//! exposes one `phyz_train_host_*` loop per kernel. Buffers are plain `Vec`s.
//! This is the reference harness for the training port: it runs the exact
//! text NVRTC will compile, on any machine, so the parity test against the
//! f64 CPU optimizer is a CI test and not a pod errand.

use super::train::{AdamCfg, NetDims, TrainBackend};

unsafe extern "C" {
    fn phyz_train_host_fwd(
        n_threads: u32,
        b: u32,
        n_in: u32,
        n_h: u32,
        n_out: u32,
        w: *const f32,
        x: *const f32,
        idx: *const u32,
        h1: *mut f32,
        h2: *mut f32,
        out: *mut f32,
    );
    fn phyz_train_host_ppo_grad(
        n_threads: u32,
        b: u32,
        n_out: u32,
        clip: f32,
        entropy_coef: f32,
        mean: *const f32,
        logstd: *const f32,
        act: *const f32,
        logp_old: *const f32,
        adv: *const f32,
        idx: *const u32,
        dmean: *mut f32,
        dlogstd: *mut f32,
        stats: *mut f32,
    );
    fn phyz_train_host_value_grad(
        n_threads: u32,
        b: u32,
        vdelta: f32,
        values: *const f32,
        ret: *const f32,
        idx: *const u32,
        dv: *mut f32,
        stats: *mut f32,
    );
    fn phyz_train_host_bwd_hidden(
        n_threads: u32,
        b: u32,
        n_out: u32,
        n_h: u32,
        dout: *const f32,
        w: *const f32,
        hact: *const f32,
        dh: *mut f32,
    );
    fn phyz_train_host_grad_w_idx(
        n_threads: u32,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: *const f32,
        x: *const f32,
        idx: *const u32,
        gw: *mut f32,
        gw_off: u32,
    );
    fn phyz_train_host_grad_w(
        n_threads: u32,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: *const f32,
        x: *const f32,
        gw: *mut f32,
        gw_off: u32,
    );
    fn phyz_train_host_grad_b(
        n_threads: u32,
        b: u32,
        n_out: u32,
        dout: *const f32,
        gb: *mut f32,
        gb_off: u32,
    );
    fn phyz_train_host_adam(
        n_threads: u32,
        n: u32,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        bc1: f64,
        bc2: f64,
        g: *const f32,
        m: *mut f64,
        v: *mut f64,
        p: *mut f64,
        w: *mut f32,
    );
    fn phyz_train_host_reduce_stats(n_threads: u32, b: u32, stats: *const f32, out: *mut f32);
}

/// Runs the training kernels on the host CPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostTrainBackend;

fn range<T: Copy>(buf: &[T], start: usize, len: usize) -> Result<Vec<T>, String> {
    if start + len > buf.len() {
        return Err(format!(
            "download of {len} from {start} of a {}-element buffer",
            buf.len()
        ));
    }
    Ok(buf[start..start + len].to_vec())
}

fn put<T: Copy>(buf: &mut [T], data: &[T]) -> Result<(), String> {
    if data.len() > buf.len() {
        return Err(format!(
            "upload of {} into a {}-element buffer",
            data.len(),
            buf.len()
        ));
    }
    buf[..data.len()].copy_from_slice(data);
    Ok(())
}

impl TrainBackend for HostTrainBackend {
    type F32 = Vec<f32>;
    type F64 = Vec<f64>;
    type U32 = Vec<u32>;

    fn device_name(&self) -> String {
        "host C++ (phyz_train.cu)".into()
    }

    fn alloc_f32(&self, len: usize) -> Result<Vec<f32>, String> {
        Ok(vec![0.0; len])
    }
    fn alloc_f64(&self, len: usize) -> Result<Vec<f64>, String> {
        Ok(vec![0.0; len])
    }
    fn alloc_u32(&self, len: usize) -> Result<Vec<u32>, String> {
        Ok(vec![0; len])
    }

    fn upload_f32(&self, buf: &mut Vec<f32>, data: &[f32]) -> Result<(), String> {
        put(buf, data)
    }
    fn upload_f64(&self, buf: &mut Vec<f64>, data: &[f64]) -> Result<(), String> {
        put(buf, data)
    }
    fn upload_u32(&self, buf: &mut Vec<u32>, data: &[u32]) -> Result<(), String> {
        put(buf, data)
    }

    fn download_f32(&self, buf: &Vec<f32>, start: usize, len: usize) -> Result<Vec<f32>, String> {
        range(buf, start, len)
    }
    fn download_f64(&self, buf: &Vec<f64>, start: usize, len: usize) -> Result<Vec<f64>, String> {
        range(buf, start, len)
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn launch_fwd(
        &self,
        b: u32,
        dims: NetDims,
        w: &Vec<f32>,
        x: &Vec<f32>,
        idx: &Vec<u32>,
        h1: &mut Vec<f32>,
        h2: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: every buffer is at least the size the kernel indexes, by
        // the pipeline's allocation; the loop is the kernel's own grid.
        unsafe {
            phyz_train_host_fwd(
                b,
                b,
                dims.n_in as u32,
                dims.n_h as u32,
                dims.n_out as u32,
                w.as_ptr(),
                x.as_ptr(),
                idx.as_ptr(),
                h1.as_mut_ptr(),
                h2.as_mut_ptr(),
                out.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_ppo_grad(
        &self,
        b: u32,
        n_out: u32,
        clip: f32,
        entropy_coef: f32,
        mean: &Vec<f32>,
        logstd: &Vec<f32>,
        logstd_off: u32,
        act: &Vec<f32>,
        logp_old: &Vec<f32>,
        adv: &Vec<f32>,
        idx: &Vec<u32>,
        dmean: &mut Vec<f32>,
        dlogstd: &mut Vec<f32>,
        stats: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as above; `logstd_off` is inside `logstd` by construction.
        unsafe {
            phyz_train_host_ppo_grad(
                b,
                b,
                n_out,
                clip,
                entropy_coef,
                mean.as_ptr(),
                logstd.as_ptr().add(logstd_off as usize),
                act.as_ptr(),
                logp_old.as_ptr(),
                adv.as_ptr(),
                idx.as_ptr(),
                dmean.as_mut_ptr(),
                dlogstd.as_mut_ptr(),
                stats.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_value_grad(
        &self,
        b: u32,
        vdelta: f32,
        values: &Vec<f32>,
        ret: &Vec<f32>,
        idx: &Vec<u32>,
        dv: &mut Vec<f32>,
        stats: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            phyz_train_host_value_grad(
                b,
                b,
                vdelta,
                values.as_ptr(),
                ret.as_ptr(),
                idx.as_ptr(),
                dv.as_mut_ptr(),
                stats.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_bwd_hidden(
        &self,
        b: u32,
        n_out: u32,
        n_h: u32,
        dout: &Vec<f32>,
        w: &Vec<f32>,
        w_off: u32,
        hact: &Vec<f32>,
        dh: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as above; `w_off` is a layer offset inside `w`.
        unsafe {
            phyz_train_host_bwd_hidden(
                b * n_h,
                b,
                n_out,
                n_h,
                dout.as_ptr(),
                w.as_ptr().add(w_off as usize),
                hact.as_ptr(),
                dh.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_grad_w_idx(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &Vec<f32>,
        x: &Vec<f32>,
        idx: &Vec<u32>,
        gw: &mut Vec<f32>,
        gw_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            phyz_train_host_grad_w_idx(
                n_out * n_in,
                b,
                n_out,
                n_in,
                dout.as_ptr(),
                x.as_ptr(),
                idx.as_ptr(),
                gw.as_mut_ptr(),
                gw_off,
            );
        }
        Ok(())
    }

    fn launch_grad_w(
        &self,
        b: u32,
        n_out: u32,
        n_in: u32,
        dout: &Vec<f32>,
        x: &Vec<f32>,
        gw: &mut Vec<f32>,
        gw_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            phyz_train_host_grad_w(
                n_out * n_in,
                b,
                n_out,
                n_in,
                dout.as_ptr(),
                x.as_ptr(),
                gw.as_mut_ptr(),
                gw_off,
            );
        }
        Ok(())
    }

    fn launch_grad_b(
        &self,
        b: u32,
        n_out: u32,
        dout: &Vec<f32>,
        gb: &mut Vec<f32>,
        gb_off: u32,
    ) -> Result<(), String> {
        // SAFETY: as above.
        unsafe {
            phyz_train_host_grad_b(n_out, b, n_out, dout.as_ptr(), gb.as_mut_ptr(), gb_off);
        }
        Ok(())
    }

    fn launch_adam(
        &self,
        n: u32,
        cfg: AdamCfg,
        bc1: f64,
        bc2: f64,
        g: &Vec<f32>,
        m: &mut Vec<f64>,
        v: &mut Vec<f64>,
        p: &mut Vec<f64>,
        w: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: every buffer is `n` long by the pipeline's allocation.
        unsafe {
            phyz_train_host_adam(
                n,
                n,
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.epsilon,
                cfg.weight_decay,
                bc1,
                bc2,
                g.as_ptr(),
                m.as_mut_ptr(),
                v.as_mut_ptr(),
                p.as_mut_ptr(),
                w.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_reduce_stats(
        &self,
        b: u32,
        stats: &Vec<f32>,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: `out` is four floats, which is the kernel's whole grid.
        unsafe {
            phyz_train_host_reduce_stats(4, b, stats.as_ptr(), out.as_mut_ptr());
        }
        Ok(())
    }
}
