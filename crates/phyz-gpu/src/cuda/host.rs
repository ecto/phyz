//! The CUDA C kernels, compiled as host C++ and walked serially.
//!
//! `build.rs` compiles `cuda/phyz_kernels.cu` with the system C++ compiler
//! (no CUDA toolkit involved) and exposes one `phyz_host_*` loop per pass.
//! Buffers are plain `Vec<f32>`. This is the reference harness for the CUDA
//! port: it runs the exact text NVRTC will compile against phyz's CPU
//! dynamics on any machine.

use super::{
    AbaArgs, ContactArgs, FkArgs, IntegrateArgs, KernelBackend, ObsArgs, PdArgs, PolicyArgs,
};

unsafe extern "C" {
    fn phyz_host_pd(
        n_threads: u32,
        nworld: u32,
        nq: u32,
        nv: u32,
        n_dofs: u32,
        dofs: *const f32,
        q: *const f32,
        v: *const f32,
        targets: *const f32,
        ctrl: *mut f32,
    );
    fn phyz_host_contact(
        n_threads: u32,
        cparams: *const f32,
        bodies: *const f32,
        geometry: *const f32,
        q: *const f32,
        v: *const f32,
        ext_forces: *mut f32,
        contact_state: *mut f32,
        hf_heights: *const f32,
        qdd: *const f32,
    );
    fn phyz_host_aba(
        n_threads: u32,
        nworld: u32,
        nv: u32,
        dt: f32,
        nbodies: u32,
        gx: f32,
        gy: f32,
        gz: f32,
        bodies: *const f32,
        q: *const f32,
        v: *const f32,
        ctrl: *const f32,
        qdd: *mut f32,
        ext_forces: *const f32,
    );
    fn phyz_host_integrate(
        n_threads: u32,
        nworld: u32,
        nv: u32,
        dt: f32,
        nbodies: u32,
        q: *mut f32,
        v: *mut f32,
        qdd: *const f32,
        bodies: *const f32,
    );
    fn phyz_host_fk(
        n_threads: u32,
        nworld: u32,
        nv: u32,
        nbodies: u32,
        bodies: *const f32,
        q: *const f32,
        v: *const f32,
        xforms: *mut f32,
    );
    fn phyz_host_obs(
        n_threads: u32,
        nworld: u32,
        nq: u32,
        nv: u32,
        nbodies: u32,
        n_in: u32,
        obs_off: u32,
        ops: *const f32,
        q: *const f32,
        v: *const f32,
        xforms: *const f32,
        obs: *mut f32,
    );
    fn phyz_host_policy(
        n_threads: u32,
        nworld: u32,
        n_in: u32,
        n_h: u32,
        n_out: u32,
        n_dofs: u32,
        act_clamp: f32,
        has_clamp_slots: u32,
        rho: f32,
        obs_off: u32,
        out_off: u32,
        weights: *const f32,
        stdv: *const f32,
        in_noise: *const f32,
        obs: *mut f32,
        rng: *mut f32,
        z: *mut f32,
        act_slots: *const f32,
        act_clamp_slots: *const f32,
        base_targets: *const f32,
        targets: *mut f32,
        out: *mut f32,
    );
}

/// Runs the CUDA C kernels on the host CPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostBackend;

impl KernelBackend for HostBackend {
    type Buffer = Vec<f32>;
    /// Graphs are a passthrough here: `supports_graphs` stays at its `false`
    /// default, so `BatchSim` executes the same call sequence directly and
    /// the parity harness compares the arithmetic the device path replays.
    type Graph = ();

    fn device_name(&self) -> String {
        "host (CUDA C compiled as C++)".into()
    }

    fn alloc(&self, len: usize) -> Result<Vec<f32>, String> {
        Ok(vec![0.0; len])
    }

    fn upload(&self, buf: &mut Vec<f32>, data: &[f32]) -> Result<(), String> {
        if data.len() > buf.len() {
            return Err(format!(
                "upload of {} floats into a {}-float buffer",
                data.len(),
                buf.len()
            ));
        }
        buf[..data.len()].copy_from_slice(data);
        Ok(())
    }

    fn download(&self, buf: &Vec<f32>) -> Result<Vec<f32>, String> {
        Ok(buf.clone())
    }

    fn synchronize(&self) -> Result<(), String> {
        Ok(())
    }

    fn download_range(&self, buf: &Vec<f32>, start: usize, len: usize) -> Result<Vec<f32>, String> {
        if start + len > buf.len() {
            return Err(format!(
                "download of {start}..{} from a {}-float buffer",
                start + len,
                buf.len()
            ));
        }
        Ok(buf[start..start + len].to_vec())
    }

    fn copy(&self, src: &Vec<f32>, dst: &mut Vec<f32>, dst_offset: usize) -> Result<(), String> {
        if dst_offset + src.len() > dst.len() {
            return Err(format!(
                "copy of {} floats at {dst_offset} into a {}-float buffer",
                src.len(),
                dst.len()
            ));
        }
        dst[dst_offset..dst_offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn launch_pd(
        &self,
        a: PdArgs,
        dofs: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        targets: &Vec<f32>,
        ctrl: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: every pointer is a live Vec of at least the length the
        // kernel indexes (sized by BatchSim from the same nworld/nq/nv).
        unsafe {
            phyz_host_pd(
                a.nworld * a.n_dofs,
                a.nworld,
                a.nq,
                a.nv,
                a.n_dofs,
                dofs.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                targets.as_ptr(),
                ctrl.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_contact(
        &self,
        a: ContactArgs,
        cparams: &Vec<f32>,
        bodies: &Vec<f32>,
        geometry: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        ext_forces: &mut Vec<f32>,
        contact_state: &mut Vec<f32>,
        hf_heights: &Vec<f32>,
        qdd: &Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_host_contact`.
        unsafe {
            phyz_host_contact(
                a.nworld,
                cparams.as_ptr(),
                bodies.as_ptr(),
                geometry.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                ext_forces.as_mut_ptr(),
                contact_state.as_mut_ptr(),
                hf_heights.as_ptr(),
                qdd.as_ptr(),
            );
        }
        Ok(())
    }

    fn launch_aba(
        &self,
        a: AbaArgs,
        bodies: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        ctrl: &Vec<f32>,
        qdd: &mut Vec<f32>,
        ext_forces: &Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd.
        unsafe {
            phyz_host_aba(
                a.nworld,
                a.nworld,
                a.nv,
                a.dt,
                a.nbodies,
                a.gx,
                a.gy,
                a.gz,
                bodies.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                ctrl.as_ptr(),
                qdd.as_mut_ptr(),
                ext_forces.as_ptr(),
            );
        }
        Ok(())
    }

    fn launch_integrate(
        &self,
        a: IntegrateArgs,
        q: &mut Vec<f32>,
        v: &mut Vec<f32>,
        qdd: &Vec<f32>,
        bodies: &Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd.
        unsafe {
            phyz_host_integrate(
                a.nworld * a.nbodies,
                a.nworld,
                a.nv,
                a.dt,
                a.nbodies,
                q.as_mut_ptr(),
                v.as_mut_ptr(),
                qdd.as_ptr(),
                bodies.as_ptr(),
            );
        }
        Ok(())
    }

    fn launch_fk(
        &self,
        a: FkArgs,
        bodies: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        xforms: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_host_fk`.
        unsafe {
            phyz_host_fk(
                a.nworld,
                a.nworld,
                a.nv,
                a.nbodies,
                bodies.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                xforms.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_obs(
        &self,
        a: ObsArgs,
        ops: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        xforms: &Vec<f32>,
        obs: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_host_obs`.
        unsafe {
            phyz_host_obs(
                a.nworld,
                a.nworld,
                a.nq,
                a.nv,
                a.nbodies,
                a.n_in,
                a.obs_off,
                ops.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                xforms.as_ptr(),
                obs.as_mut_ptr(),
            );
        }
        Ok(())
    }

    fn launch_policy(
        &self,
        a: PolicyArgs,
        weights: &Vec<f32>,
        stdv: &Vec<f32>,
        in_noise: &Vec<f32>,
        obs: &mut Vec<f32>,
        rng: &mut Vec<f32>,
        z: &mut Vec<f32>,
        act_slots: &Vec<f32>,
        act_clamp_slots: &Vec<f32>,
        base_targets: &Vec<f32>,
        targets: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd, against `phyz_host_policy`.
        unsafe {
            phyz_host_policy(
                a.nworld,
                a.nworld,
                a.n_in,
                a.n_h,
                a.n_out,
                a.n_dofs,
                a.act_clamp,
                a.has_clamp_slots,
                a.rho,
                a.obs_off,
                a.out_off,
                weights.as_ptr(),
                stdv.as_ptr(),
                in_noise.as_ptr(),
                obs.as_mut_ptr(),
                rng.as_mut_ptr(),
                z.as_mut_ptr(),
                act_slots.as_ptr(),
                act_clamp_slots.as_ptr(),
                base_targets.as_ptr(),
                targets.as_mut_ptr(),
                out.as_mut_ptr(),
            );
        }
        Ok(())
    }
}
