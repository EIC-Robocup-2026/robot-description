# Robot Description

This repository contains the Universal Robot Description Format (URDF) files for Our Walkie robot, intended for use in simulation environments, along with the necessary transformation data.

## Simulation Compatibility: Collision & Mesh Inversion Notice

### Why Negative Scale Reflection (`*-1` / Inverted Scale) Fails in Physics Simulators

In the upstream OpenArm (and common ROS URDF patterns), symmetry across left/right arms or gripper fingers is often implemented by applying a negative scale along an axis (e.g., `scale="0.001 ${0.001 * reflect} 0.001"` where `reflect = -1`, or `scale="0.001 -0.001 0.001"`).

While basic visualizers like RViz allow negative scaling matrices, **this method causes severe rendering and physics cooking failures across modern physics engines and simulation platforms** (especially PhysX-based simulators).

In this repository, all mirrored links and fingers use dedicated pre-mirrored meshes (e.g., `*_symp_inv.stl`, `*_inv.dae`, `finger_inv.stl`) with strictly **positive scales** (`scale="0.001 0.001 0.001"`).

---

### Platform Breakdown & Failure Symptoms

| Simulator / Platform | Physics Engine | Behavior with Negative Scale Reflection (`reflect = -1`) |
| :--- | :--- | :--- |
| **SAPIEN** | NVIDIA PhysX (4 / 5) | **Import Error / Crash**: Fails during URDF import when cooking convex collision meshes (`[PhysX error] Failed to create convex mesh` / assertion failure during mesh cooking), preventing the model from loading. |
| **NVIDIA Isaac Sim / Omniverse** | NVIDIA PhysX (USD PhysX) | **Broken Colliders & Missing Visuals**: Mesh disappears completely in RTX render due to inverted winding / back-face culling ("visual is gone"). PhysX collision cooking produces distorted, inverted, or ghost colliders ("weird collider") or rejects cooking entirely. |
| **O3DE (Open 3D Engine)** | NVIDIA PhysX Gem | **Physics Explosion ("Scaling Boom / Rotation Boom")**: Transform matrix decomposition fails because reflections ($\det(M) < 0$) cannot be represented by unit quaternions ($SO(3)$). Causes scale oscillation, 180° rotation flipping, and solver instability resulting in exploding joints and meshes. |
| **Gazebo / Ignition (GZ)** | DART / Bullet / ODE | **Contact & Normal Inversion**: Flipped face normals cause inverted collision contacts, penetrating meshes, and erratic inter-penetration forces. |
| **MuJoCo** | MuJoCo MJCF / Compiler | **Not Supported**: URDF importer rejects negative mesh scales; requires positive scales and proper spatial orientations. |

---

### Technical Deep Dive: Why Does It Break?

1. **Chirality Inversion & Polygon Winding Order (Clockwise vs. Counter-Clockwise)**:
   - Applying a negative scale along a single axis results in a transformation matrix with a negative determinant ($\det(M) < 0$), switching the coordinate frame chirality from right-handed to left-handed.
   - This inverts the polygon vertex winding order (counter-clockwise faces become clockwise).
   - **Visual Rendering**: Graphics pipelines (OpenGL, Vulkan, DirectX, RTX) rely on winding order for back-face culling. Front faces appear as back faces and are culled out, rendering the mesh invisible.
   - **Physics / Collision**: Convex hull cooking algorithms (e.g., Quickhull, PhysX Cooking) rely on counter-clockwise winding to compute outward-facing face normals and half-space bounding planes. Inverted winding results in inward-pointing normals, negative volumes, or degenerate collision hulls.

2. **NVIDIA PhysX Architectural Constraints (`PxCooking` & `PxMeshScale`)**:
   - NVIDIA PhysX strictly models rigid bodies using the Special Euclidean group $SE(3)$ (rotations in $SO(3)$ + translation vectors).
   - Convex mesh cooking (`PxConvexMeshDesc`) and triangle mesh cooking (`PxTriangleMeshDesc`) require positive volumes and valid surface orientation.
   - Negative scaling parameters inside `PxMeshScale` violate PhysX internal invariants, causing mesh cooker failure, empty hull fallbacks, or physics engine assertion crashes.

3. **Quaternion Incompatibility with Reflection Matrices ($O(3) \setminus SO(3)$)**:
   - Modern simulation frameworks and game engines (such as O3DE, Unreal, Unity) store 3D rotation using unit quaternions ($\mathbb{H}$), which can only represent pure proper rotations with determinant $+1$.
   - A reflection matrix (improper rotation with $\det = -1$) cannot be factored into a quaternion and positive scale without sign ambiguity.
   - During runtime transform propagation and physics updates, the engine repeatedly attempts polar/QR matrix decomposition. This creates numerical instability, flipping quaternions by 180° or alternating sign values every frame, causing visual and collision transforms to "boom" (explode/jitter violently).

---

### Solution Implemented in This Repository

1. **Pre-mirrored Geometry**: Mirrored links are created offline (via Blender / CAD) by mirroring the geometry, applying all transforms to identity, recalculating outward face normals, and saving them as explicit meshes (e.g., `*_inv.dae` for visual and `*_symp_inv.stl` for collision).
2. **Uniform Positive Scale in Xacro**: All mesh tags strictly use positive scales:
   ```xml
   <!-- Correct approach: Positive scale with dynamically chosen pre-mirrored mesh -->
   <xacro:property name="visual_mesh" value="${name + '_inv.dae' if reflect == -1 else name + '.dae'}" />
   <xacro:property name="collision_mesh" value="${name + '_symp_inv.stl' if reflect == -1 else name + '_symp.stl'}" />

   <visual name="${prefix}${name}_visual">
     <geometry>
       <mesh filename="package://${description_pkg}/meshes/openarm/arm/${arm_type}/visual/${visual_mesh}" scale="0.001 0.001 0.001" />
     </geometry>
   </visual>
   <collision name="${prefix}${name}_collision">
     <geometry>
       <mesh filename="package://${description_pkg}/meshes/openarm/arm/${arm_type}/collision/${collision_mesh}" scale="0.001 0.001 0.001" />
     </geometry>
   </collision>
   ```

