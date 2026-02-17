//! Spatial bounding regions for coupling.

use phyz_math::Vec3;

/// Axis-aligned bounding box for overlap regions.
#[derive(Clone, Debug)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Check if a point is inside the bounding box.
    pub fn contains(&self, point: &Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Compute volume of the bounding box.
    pub fn volume(&self) -> f64 {
        let size = self.max - self.min;
        size.x * size.y * size.z
    }

    /// Compute center of the bounding box.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Expand the bounding box by a margin.
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            min: self.min - Vec3::new(margin, margin, margin),
            max: self.max + Vec3::new(margin, margin, margin),
        }
    }

    /// Check if two bounding boxes overlap.
    pub fn overlaps(&self, other: &BoundingBox) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Compute intersection of two bounding boxes.
    pub fn intersection(&self, other: &BoundingBox) -> Option<BoundingBox> {
        if !self.overlaps(other) {
            return None;
        }

        Some(BoundingBox {
            min: Vec3::new(
                self.min.x.max(other.min.x),
                self.min.y.max(other.min.y),
                self.min.z.max(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.min(other.max.x),
                self.max.y.min(other.max.y),
                self.max.z.min(other.max.z),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_contains() {
        let bbox = BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));

        assert!(bbox.contains(&Vec3::new(0.0, 0.0, 0.0)));
        assert!(bbox.contains(&Vec3::new(0.5, 0.5, 0.5)));
        assert!(bbox.contains(&Vec3::new(-1.0, -1.0, -1.0)));
        assert!(bbox.contains(&Vec3::new(1.0, 1.0, 1.0)));
        assert!(!bbox.contains(&Vec3::new(2.0, 0.0, 0.0)));
        assert!(!bbox.contains(&Vec3::new(0.0, 2.0, 0.0)));
    }

    #[test]
    fn test_volume() {
        let bbox = BoundingBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 3.0, 4.0));
        assert_relative_eq!(bbox.volume(), 24.0);
    }

    #[test]
    fn test_center() {
        let bbox = BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let center = bbox.center();
        assert_relative_eq!(center.x, 0.0);
        assert_relative_eq!(center.y, 0.0);
        assert_relative_eq!(center.z, 0.0);
    }

    #[test]
    fn test_expand() {
        let bbox = BoundingBox::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let expanded = bbox.expand(0.5);

        assert_relative_eq!(expanded.min.x, -1.5);
        assert_relative_eq!(expanded.max.x, 1.5);
    }

    #[test]
    fn test_overlaps() {
        let bbox1 = BoundingBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let bbox2 = BoundingBox::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));
        let bbox3 = BoundingBox::new(Vec3::new(3.0, 3.0, 3.0), Vec3::new(5.0, 5.0, 5.0));

        assert!(bbox1.overlaps(&bbox2));
        assert!(bbox2.overlaps(&bbox1));
        assert!(!bbox1.overlaps(&bbox3));
    }

    #[test]
    fn test_intersection() {
        let bbox1 = BoundingBox::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        let bbox2 = BoundingBox::new(Vec3::new(1.0, 1.0, 1.0), Vec3::new(3.0, 3.0, 3.0));

        let intersection = bbox1.intersection(&bbox2).unwrap();
        assert_relative_eq!(intersection.min.x, 1.0);
        assert_relative_eq!(intersection.max.x, 2.0);

        let bbox3 = BoundingBox::new(Vec3::new(3.0, 3.0, 3.0), Vec3::new(5.0, 5.0, 5.0));
        assert!(bbox1.intersection(&bbox3).is_none());
    }
}
