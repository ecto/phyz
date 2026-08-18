//! The CUDA C kernels, compiled as host C++ and walked serially.
//!
//! `build.rs` compiles `cuda/phyz_kernels.cu` with the system C++ compiler
//! (no CUDA toolkit involved) and exposes one `phyz_host_*` loop per pass.
//! Buffers are plain `Vec<f32>`. This is the reference harness for the CUDA
//! port: it runs the exact text NVRTC will compile against phyz's CPU
//! dynamics on any machine.

use super::{AbaArgs, ContactArgs, IntegrateArgs, KernelBackend, PdArgs};

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
        nworld: u32,
        nbodies: u32,
        nv: u32,
        ground_height: f32,
        dt: f32,
        bodies: *const f32,
        geometry: *const f32,
        q: *const f32,
        v: *const f32,
        ext_forces: *mut f32,
        contact_state: *mut f32,
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
}

/// Runs the CUDA C kernels on the host CPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct HostBackend;

impl KernelBackend for HostBackend {
    type Buffer = Vec<f32>;

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
        bodies: &Vec<f32>,
        geometry: &Vec<f32>,
        q: &Vec<f32>,
        v: &Vec<f32>,
        ext_forces: &mut Vec<f32>,
        contact_state: &mut Vec<f32>,
    ) -> Result<(), String> {
        // SAFETY: as in launch_pd.
        unsafe {
            phyz_host_contact(
                a.nworld,
                a.nworld,
                a.nbodies,
                a.nv,
                a.ground_height,
                a.dt,
                bodies.as_ptr(),
                geometry.as_ptr(),
                q.as_ptr(),
                v.as_ptr(),
                ext_forces.as_mut_ptr(),
                contact_state.as_mut_ptr(),
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
}
