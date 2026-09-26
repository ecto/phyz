//! MJCF geom contact attributes become `Body::material`.
//!
//! MJCF is the only importable format that expresses contact materials at all
//! — URDF has no standard element for friction and `phyz-urdf` reads none, by
//! design and documented there. So this is where a robot description can
//! carry "the soles are grippy and the wheels are not" into the sim.

use phyz_mjcf::MjcfLoader;
use phyz_model::ContactMaterial;

fn model_from(xml: &str) -> phyz_model::Model {
    MjcfLoader::from_xml_str(xml).expect("parse").build_model()
}

fn material_of(model: &phyz_model::Model, name: &str) -> Option<ContactMaterial> {
    let i = model.body_index(name).unwrap_or_else(|| {
        panic!(
            "no body {name}; have {:?}",
            model.bodies.iter().map(|b| &b.name).collect::<Vec<_>>()
        )
    });
    model.bodies[i].material.clone()
}

/// The headline: `friction`, `solref`, `solimp` and `margin` on a geom reach
/// the body's material, and a geom that names none of them leaves the body on
/// the scene material.
#[test]
fn geom_contact_attributes_become_a_body_material() {
    let model = model_from(
        r#"
        <mujoco>
          <worldbody>
            <body name="sole" pos="0 0 1">
              <freejoint/>
              <geom type="box" size="0.1 0.05 0.01" friction="1.5 0.01 0.001"/>
            </body>
            <body name="wheel" pos="1 0 1">
              <freejoint/>
              <geom type="sphere" size="0.03" friction="0.75"
                    solref="0.01 0.8" solimp="0.8 0.99 0.002" margin="0.005"/>
            </body>
            <body name="plain" pos="2 0 1">
              <freejoint/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    );

    // Only the sliding component of MuJoCo's `slide spin roll` is read; phyz's
    // cone has no torsional term to put the other two in.
    let sole = material_of(&model, "sole").expect("sole names friction");
    assert_eq!(sole.friction, 1.5);
    // Unnamed fields keep phyz's defaults, not MuJoCo's.
    assert_eq!(sole.margin, ContactMaterial::default().margin);
    assert_eq!(
        sole.solref.timeconst,
        ContactMaterial::default().solref.timeconst
    );

    let wheel = material_of(&model, "wheel").expect("wheel names several");
    assert_eq!(wheel.friction, 0.75);
    assert_eq!(wheel.margin, 0.005);
    assert_eq!(wheel.solref.timeconst, 0.01);
    assert_eq!(wheel.solref.dampratio, 0.8);
    assert_eq!(wheel.solimp.dmin, 0.8);
    assert_eq!(wheel.solimp.dmax, 0.99);
    assert_eq!(wheel.solimp.width, 0.002);
    // A three-element solimp keeps the default midpoint and power, as in
    // MuJoCo's own shorthand.
    assert_eq!(
        wheel.solimp.midpoint,
        ContactMaterial::default().solimp.midpoint
    );
    assert_eq!(wheel.solimp.power, ContactMaterial::default().solimp.power);

    // The load-bearing negative: a geom that says nothing must not import
    // MuJoCo's default friction of 1.0, because phyz's default is 0.5 and
    // every existing model would silently get grippier.
    assert!(
        material_of(&model, "plain").is_none(),
        "a geom with no contact attributes must leave the body on the scene material"
    );
}

/// A `<default>` class counts as the geom naming the attribute — that is how
/// real MJCF robots express "all the collision geoms are this material".
#[test]
fn defaults_classes_supply_the_material() {
    let model = model_from(
        r#"
        <mujoco>
          <default>
            <default class="foot">
              <geom friction="1.4"/>
            </default>
          </default>
          <worldbody>
            <body name="foot" pos="0 0 1">
              <freejoint/>
              <geom class="foot" type="box" size="0.1 0.05 0.01"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    );
    assert_eq!(
        material_of(&model, "foot").expect("from class").friction,
        1.4
    );
}

/// Several geoms on one body fold into one material by the same
/// `ContactMaterial::combine` rule a contacting pair uses — max friction —
/// and geoms that named nothing are skipped rather than averaging the
/// explicit one back down.
#[test]
fn several_geoms_fold_into_one_body_material() {
    let model = model_from(
        r#"
        <mujoco>
          <worldbody>
            <body name="shoe" pos="0 0 1">
              <freejoint/>
              <geom name="sole" type="box" size="0.1 0.05 0.01" friction="1.5"/>
              <geom name="heel" type="box" size="0.03 0.05 0.02" pos="-0.08 0 0" friction="0.9"/>
              <geom name="upper" type="box" size="0.09 0.05 0.03" pos="0 0 0.04"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    );
    let shoe = material_of(&model, "shoe").expect("two geoms name friction");
    assert_eq!(
        shoe.friction, 1.5,
        "max, not mean, and `upper` must not dilute it"
    );
}

/// Visual-only geoms (`contype=0 conaffinity=0`) contribute inertia but never
/// contact, so their attributes must not reach the contact material.
#[test]
fn visual_geoms_do_not_contribute_a_contact_material() {
    let model = model_from(
        r#"
        <mujoco>
          <worldbody>
            <body name="part" pos="0 0 1">
              <freejoint/>
              <geom type="box" size="0.1 0.1 0.1" friction="0.3"/>
              <geom type="box" size="0.2 0.2 0.2" friction="2.0"
                    contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        "#,
    );
    let part = material_of(&model, "part").expect("collision geom names friction");
    assert_eq!(part.friction, 0.3, "the visual geom's 2.0 must not win");
}
