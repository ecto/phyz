//! MJCF joint/actuator attributes must reach the dynamics, not just the model.

use phyz_mjcf::MjcfLoader;
use phyz_rigid::aba;

const GEARED: &str = r#"
<mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="link1" pos="0 0 0">
            <inertial mass="1.0" diaginertia="0.25 0.25 0.25"/>
            <joint name="j1" type="hinge" axis="0 0 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor name="m1" joint="j1" gear="100" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"#;

/// Angular acceleration for a control value, with gravity zeroed in the XML.
fn qdd_for_ctrl(ctrl: f64) -> f64 {
    let model = MjcfLoader::from_xml_str(GEARED).unwrap().build_model();
    let mut state = model.default_state();
    state.ctrl[0] = ctrl;
    aba(&model, &state)[0]
}

#[test]
fn geared_actuator_produces_xml_specified_torque() {
    // gear=100, ctrl=0.5 -> 50 N·m on a 0.25 kg·m² link.
    let expected = 50.0 / 0.25;
    let qdd = qdd_for_ctrl(0.5);
    assert!(
        (qdd - expected).abs() < 1e-9,
        "qdd = {}, expected {}",
        qdd,
        expected
    );
}

#[test]
fn geared_actuator_clamps_to_ctrlrange() {
    // ctrl=7.5 clamps to 1.0 -> 100 N·m.
    let expected = 100.0 / 0.25;
    let qdd = qdd_for_ctrl(7.5);
    assert!(
        (qdd - expected).abs() < 1e-9,
        "qdd = {}, expected {}",
        qdd,
        expected
    );
    assert!((qdd_for_ctrl(7.5) - qdd_for_ctrl(1.0)).abs() < 1e-12);
    assert!((qdd_for_ctrl(-7.5) - qdd_for_ctrl(-1.0)).abs() < 1e-12);
}

#[test]
fn joint_attributes_reach_the_model() {
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="link1" pos="0 0 0">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="j1" type="hinge" axis="0 0 1" range="-0.5 0.5"
                       damping="0.1" armature="0.05" stiffness="3"
                       springref="0.2" frictionloss="0.7"/>
            </body>
        </worldbody>
    </mujoco>
    "#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    let j = &model.joints[0];
    assert_eq!(j.limits, Some([-0.5, 0.5]));
    assert!((j.damping - 0.1).abs() < 1e-12);
    assert!((j.armature - 0.05).abs() < 1e-12);
    assert!((j.stiffness - 3.0).abs() < 1e-12);
    assert!((j.spring_ref - 0.2).abs() < 1e-12);
    assert!((j.friction_loss - 0.7).abs() < 1e-12);
}

#[test]
fn limited_false_disables_range() {
    let mjcf = r#"
    <mujoco>
        <worldbody>
            <body name="link1" pos="0 0 0">
                <inertial mass="1.0" diaginertia="0.1 0.1 0.1"/>
                <joint name="j1" type="hinge" axis="0 0 1" range="-0.5 0.5" limited="false"/>
            </body>
        </worldbody>
    </mujoco>
    "#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    assert_eq!(model.joints[0].limits, None);
}

#[test]
fn mjcf_joint_limit_stops_the_link() {
    let mjcf = r#"
    <mujoco>
        <option gravity="0 -9.81 0"/>
        <worldbody>
            <body name="link1" pos="0 0 0">
                <inertial mass="1.0" pos="0 -0.5 0" diaginertia="0.083 0.083 0.083"/>
                <joint name="j1" type="hinge" axis="0 0 1" range="-0.3 0.3"/>
            </body>
        </worldbody>
    </mujoco>
    "#;
    let model = MjcfLoader::from_xml_str(mjcf).unwrap().build_model();
    let mut state = model.default_state();
    state.q[0] = 0.3;
    state.v[0] = 1.0;

    let dt = 1e-4;
    for _ in 0..20_000 {
        let qdd = aba(&model, &state);
        state.v[0] += dt * qdd[0];
        state.q[0] += dt * state.v[0];
        assert!(state.q[0].is_finite(), "diverged");
    }
    assert!(
        state.q[0] > -0.4 && state.q[0] < 0.4,
        "left the limit band: q = {}",
        state.q[0]
    );
}
