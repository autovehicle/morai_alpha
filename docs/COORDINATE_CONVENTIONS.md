# Coordinate conventions and validation gate

## Provisional source assumption

The extracted mathematical helper supports the convention currently implied by
the old renderer:

- world yaw zero points along world +X;
- positive world yaw rotates counter-clockwise toward world +Y;
- object heading uses the same reference and sign as Ego yaw;
- size X is length and size Y is width.

These are assumptions, not verified facts. Dataset GT conversion uses a strict
verification gate by default.

## Target Dataset frame

- x: Ego forward
- y: Ego left
- z: up
- yaw: radians, positive counter-clockwise
- position/size: metres
- normalized velocity: m/s where a normalized value is produced
- time: nanoseconds

## Competition-MORAI validation checklist

Before setting `coordinate_convention.verified: true`:

1. Place Ego at a known world pose and verify which world axis is forward at
   yaw 0, +90, -90, and 180 degrees.
2. Place one object directly ahead, left, right, and behind; confirm converted
   signs and BEV overlay positions.
3. Rotate an asymmetric/arrow-shaped object and confirm heading zero, sign, and
   object forward axis.
4. Confirm whether Object position z is centre, base, or another reference.
5. Confirm `size_x/y/z` correspond to length/width/height and are full extents.
6. Confirm LiDAR x/y/z axes and origin relative to Ego.
7. Record Camera intrinsics, distortion model, image orientation, and each
   camera-to-Ego extrinsic.
8. Reproject LiDAR/object points into images and inspect several known poses.
9. Save the verified convention and calibration version with every sequence.

The old `bev_render.py` remains a visualization reference only. In particular,
its object-corner comments disagree on whether positive local Y is left or
right, so its comments are not a coordinate contract.
