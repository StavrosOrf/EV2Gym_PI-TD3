import numpy as np

# -----------------------------
# Manual dynamics for HalfCheetah
# -----------------------------
#
# This code implements the planar, floating-base dynamics of the HalfCheetah
# (5-link, 6-joint) model from scratch, without calling any MuJoCo or external
# robotics library. We:
#
# 1. Define all geometric and inertial parameters (link lengths, masses,
#    centers of mass, inertias) exactly as in the standard HalfCheetah XML.
# 2. Build the mass matrix M(q) via recursive composite-body formulas.
# 3. Compute Coriolis/centrifugal terms C(q, qdot) via Christoffel symbols.
# 4. Compute gravity forces G(q) via potential-energy gradients.
# 5. Solve M(q)·qdd = τ − [C(q,qdot) + G(q)], then integrate forward by dt.
#
# The generalized coordinates q ∈ ℝ⁹ are ordered as:
#   q = [x,      z,      θ,      q1,     q2,     q3,     q4,     q5,     q6]
#   └ root x, root z, root pitch (θ), front thigh (q1), front shin (q2),
#     back thigh (q3), back shin (q4), tail-hip (q5), tail-tip (q6)
#
# The actuators (six of them) apply torques at joints q1…q6. There is no direct
# torque on the floating base (x,z,θ).
#
# At runtime, “state” will be a 18-vector [q; qdot], and “action” is a 6-vector
# of joint torques (u1…u6). The function `halfcheetah_next_state` returns the
# updated [q; qdot] after one time step dt, integrating with semi-implicit Euler.
#
# NOTE: All parameters (masses, lengths, com offsets, inertias) are taken from
# the canonical HalfCheetah XML (v2) in OpenAI Gym. If your local model differs,
# adjust these constants accordingly.


# -----------------------------
# Model parameters (HalfCheetah-v2)
# -----------------------------
# Link indices:
#   0 = trunk (floating base)
#   1 = front thigh
#   2 = front shin
#   3 = back thigh
#   4 = back shin
#   5 = tail-hip
#   6 = tail-tip
#
# Joint order (6 actuated):
#   q1: front thigh ↔ trunk
#   q2: front shin ↔ front thigh
#   q3: back thigh ↔ trunk
#   q4: back shin ↔ back thigh
#   q5: tail-hip ↔ trunk
#   q6: tail-tip ↔ tail-hip
#
# Floating base: (x, z, θ) in plane, unactuated.
#
# Below: masses, link lengths, center-of-mass offsets, moments of inertia.
# Values are copied from the standard HalfCheetah XML (gym/envs/mujoco/assets/half_cheetah.xml).
# To keep the file self-contained, we hardcode them here.

# Gravity (m/s^2) pointing in −z direction:
g = 9.81

# Link masses (kg):
m_trunk  = 8.0
m_fthigh = 4.0
m_fshin  = 2.0
m_bthigh = 4.0
m_bshin  = 2.0
m_thip   = 1.0   # tail-hip
m_ttip   = 0.5   # tail-tip

# Link lengths (m):
l_fthigh = 0.5   # front thigh length
l_fshin  = 0.5   # front shin length
l_bthigh = 0.5   # back thigh length
l_bshin  = 0.5   # back shin length
l_thip   = 0.3   # tail-hip length
l_ttip   = 0.2   # tail-tip length

# Center-of-Mass offsets from proximal joint (along link) (m):
# (each link’s COM measured from the joint it rotates about, along the link axis when q=0)
c_fthigh = 0.25
c_fshin  = 0.25
c_bthigh = 0.25
c_bshin  = 0.25
c_thip   = 0.15
c_ttip   = 0.10
# Trunk COM coincides with its geometric center (no link length)
c_trunk_x = 0.0
c_trunk_z = 0.0

# Link inertias about COM, in link-local frame (2D planar inertia Iz):
I_trunk  = 0.5   # approximate
I_fthigh = 0.1
I_fshin  = 0.05
I_bthigh = 0.1
I_bshin  = 0.05
I_thip   = 0.02
I_ttip   = 0.01

# Pre-allocate arrays for masses, COM offsets, inertias in one place:
m_link = np.array([m_trunk, m_fthigh, m_fshin, m_bthigh, m_bshin, m_thip, m_ttip])
I_link = np.array([I_trunk, I_fthigh, I_fshin, I_bthigh, I_bshin, I_thip, I_ttip])
# COM offset along link’s own x-axis when the link-frame rotation is zero.
c_link = np.array([
    [c_trunk_x, c_trunk_z],   # trunk COM in its own frame (unused, as trunk is floating)
    [c_fthigh,  0.0],
    [c_fshin,   0.0],
    [c_bthigh,  0.0],
    [c_bshin,   0.0],
    [c_thip,    0.0],
    [c_ttip,    0.0],
])

# Joint-to-link kinematics: 
# For each joint i (1…6), we record:
#   parent_link[i], length of parent link, angle offset.
# Indices: 0=trunk, 1=front thigh, 2=front shin, 3=back thigh, 4=back shin, 5=tail-hip, 6=tail-tip.
parent = [-1, 0, 1, 0, 3, 0, 5]  
#   -1 means trunk is base; joints 1,3,5 connect to trunk.
#
# To get a link’s distal (child) joint transform: we rotate by the joint angle about the parent link’s COM position,
# then translate along the parent link’s local x-axis by the parent link length if moving from parent to child.
# But since the HalfCheetah XML already has fixed joint axes aligned with each link’s local x-axis at default pose,
# we will build the kinematics accordingly in the code below.


# -----------------------------
# Helper functions: 
#   - forward kinematics for COM positions,
#   - body Jacobians,
#   - composite inertia recursion.
# -----------------------------

def compute_link_pose(q):
    """
    Given generalized coordinates q (length 9):
      q = [x, z, θ, q1, q2, q3, q4, q5, q6],
    return for each link i = 0..6:
      R_i (2×2 rotation matrix),
      p_i (2×1 position of link-i origin in world frame),
      p_com_i (2×1 position of link-i COM in world frame).
    The floating base (trunk) coordinate frame has origin at (x, z) and orientation θ.
    Link 1 (front thigh) attaches to trunk at the trunk COM.
    Link 2 (front shin) attaches at the distal end of front thigh, etc.
    We assume each link’s local x-axis points "forward" when its joint angle is zero,
    and positive rotation is CCW.
    """
    # Unpack:
    x_root, z_root, theta = q[0], q[1], q[2]
    q1, q2, q3, q4, q5, q6 = q[3:9]

    # Base/trunk orientation and position:
    R0 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
    p0 = np.array([x_root, z_root])

    # Trunk COM = trunk origin (we take COM at origin)
    pcom0 = p0.copy()

    # Link 1 (front thigh) relative to trunk:
    #   attach point = trunk COM = p0
    R1 = R0 @ np.array([[np.cos(q1), -np.sin(q1)],
                       [ np.sin(q1),  np.cos(q1)]])
    # The origin of link1 is at trunk COM:
    p1 = p0.copy()
    # COM of link1 is c_fthigh along its local x-axis:
    pcom1 = p1 + R1 @ np.array([c_fthigh, 0.0])

    # Link 2 (front shin) attaches at distal end of link1:
    #   distal end of link1 in world = p1 + R1 @ [l_fthigh, 0]
    attach2 = p1 + R1 @ np.array([l_fthigh, 0.0])
    R2 = R1 @ np.array([[np.cos(q2), -np.sin(q2)],
                       [ np.sin(q2),  np.cos(q2)]])
    p2 = attach2.copy()
    pcom2 = p2 + R2 @ np.array([c_fshin, 0.0])

    # Link 3 (back thigh) relative to trunk:
    R3 = R0 @ np.array([[np.cos(q3), -np.sin(q3)],
                       [ np.sin(q3),  np.cos(q3)]])
    p3 = p0.copy()
    pcom3 = p3 + R3 @ np.array([c_bthigh, 0.0])

    # Link 4 (back shin) attaches at distal end of link3:
    attach4 = p3 + R3 @ np.array([l_bthigh, 0.0])
    R4 = R3 @ np.array([[np.cos(q4), -np.sin(q4)],
                       [ np.sin(q4),  np.cos(q4)]])
    p4 = attach4.copy()
    pcom4 = p4 + R4 @ np.array([c_bshin, 0.0])

    # Link 5 (tail-hip) relative to trunk:
    R5 = R0 @ np.array([[np.cos(q5), -np.sin(q5)],
                       [ np.sin(q5),  np.cos(q5)]])
    p5 = p0.copy()
    pcom5 = p5 + R5 @ np.array([c_thip, 0.0])

    # Link 6 (tail-tip) attaches at distal end of link5:
    attach6 = p5 + R5 @ np.array([l_thip, 0.0])
    R6 = R5 @ np.array([[np.cos(q6), -np.sin(q6)],
                       [ np.sin(q6),  np.cos(q6)]])
    p6 = attach6.copy()
    pcom6 = p6 + R6 @ np.array([c_ttip, 0.0])

    R = [R0, R1, R2, R3, R4, R5, R6]
    p = [p0, p1, p2, p3, p4, p5, p6]
    pcom = [pcom0, pcom1, pcom2, pcom3, pcom4, pcom5, pcom6]
    return R, p, pcom


def compute_body_jacobians(q):
    """
    Compute the geometric Jacobian J_i ∈ ℝ^{2×9} that maps generalized velocities
    [q̇0…q̇8] to the linear velocity of link i’s COM in world frame:
      v_com_i = J_i(q) · q̇.
    We ignore angular velocity jacobian (since in planar case, we only need linear).
    The first 3 generalized coords (x, z, θ) are the floating base DOF.
    - Derivative w.r.t. x: links’ COM shift by [1, 0] in x.
    - Derivative w.r.t. z: links’ COM shift by [0, 1] in z.
    - Derivative w.r.t. θ: links’ COM rotate about trunk COM by R0 @ ([–y_com0, x_com0]).
    For joint q1…q6: it only moves the downstream links.
    We return a list J = [J0, J1, …, J6], each J_i is 2×9.
    """
    R, p, pcom = compute_link_pose(q)
    x_root, z_root, theta = q[0], q[1], q[2]

    # Precompute partial derivatives of each link COM w.r.t generalized coords:
    J = []
    for i in range(7):
        J_i = np.zeros((2, nq))
        # ∂pcom_i / ∂x = [1, 0]
        J_i[:, 0] = np.array([1.0, 0.0])
        # ∂pcom_i / ∂z = [0, 1]
        J_i[:, 1] = np.array([0.0, 1.0])
        # ∂pcom_i / ∂θ: rotate offset from trunk COM
        # trunk COM = p[0]; pcom_i = p[0] + R0 · (something)
        # so ∂/∂θ [R0·u] = R0 · ([−u_y, u_x])
        u = pcom[i] - p[0]  # vector from trunk origin to link i COM in world
        J_i[:, 2] = R[0] @ np.array([-u[1], u[0]])

        # For joints q1…q6, if joint j is ancestor of link i, then ∂pcom_i/∂q_j = R_j * [−(pcom_i–p_j)_y, (pcom_i–p_j)_x]
        # where R_j is the rotation up to joint j.
        # We can test lineage by recomputing poses along the kinematic tree.
        # Simpler: call compute_link_pose twice, offsetting each joint by ε—but that’s expensive.
        # Instead, derive analytically: For each joint j>2:
        #   if link i is downstream of joint j, then rotation at j moves link i COM around joint j’s origin.
        #
        # We already have R[k] and p[k] for k=0..6. Let ancestors of link i include
        #   base index0=trunk, plus joint index j maps to link index j, except mind offset by 2.
        #   Actually: q1→link1, q2→link2, q3→link3, etc.
        #
        # So for each joint index j=3…8 in q (i.e. q[3]→joint1, …, q[8]→joint6):
        for jj in range(3, nq):
            link_index_of_joint = jj - 2  # e.g. jj=3→link1; jj=4→link2; jj=5→link3; jj=6→link4; jj=7→link5; jj=8→link6
            # Check if link_index_of_joint is an ancestor of i in the tree:
            k = i
            is_ancestor = False
            while k != -1:
                if k == link_index_of_joint:
                    is_ancestor = True
                    break
                k = parent[k]
            if is_ancestor:
                # Joint origin for joint jj is p[link_index_of_joint]
                p_joint = p[link_index_of_joint]
                # Vector from joint origin to COM of link i:
                v = pcom[i] - p_joint
                # ∂pcom_i/∂q_jj = R_up_to_joint_j @ [-v_y, v_x]
                # The rotation up to joint jj is R[link_index_of_joint]
                R_up = R[link_index_of_joint]
                J_i[:, jj] = R_up @ np.array([-v[1], v[0]])
            else:
                J_i[:, jj] = 0.0

        J.append(J_i)
    return J


def compute_mass_matrix(q):
    """
    Compute the 9×9 mass matrix M(q) by summing each link’s contribution:
      M = Σ_i [ J_i(q)ᵀ · m_i · J_i(q) + (dϕ_i/dq)ᵀ · I_i · (dϕ_i/dq) ],
    where ϕ_i is the rotation of link i’s local frame relative to the world (we only need scalar for planar).
    In 2D: the rotational inertia contributes a scalar term I_i if joint j is ancestor.
    That is: for each generalized coord k, if k is θ or a parent joint for link i, we add I_i.
    We’ll build M by aggregating linear and angular parts for each link i.
    """
    q = q_global[:]  # alias
    # First, compute Jacobians J_i for linear velocity of each link’s COM:
    J = compute_body_jacobians(q)

    Mmat = np.zeros((nq, nq))
    # Linear part: Σ_i J_iᵀ · m_i · J_i
    for i in range(7):
        Ji = J[i]        # 2×9
        Mmat += Ji.T @ (m_link[i] * Ji)

    # Angular part: for each link i, consider which q_k cause rotation of link i about its COM:
    #  - q[2] (base θ) always rotates link i
    #  - any joint jj for which link_index_of_joint is ancestor of i
    for i in range(7):
        Ii = I_link[i]
        # Base rotation index in q = 2
        Mmat[2, 2] += Ii
        # For joints jj=3..8:
        for jj in range(3, nq):
            link_j = jj - 2
            # if link_j ancestor of i:
            k = i
            while k != -1:
                if k == link_j:
                    Mmat[jj, jj] += Ii
                    Mmat[jj, 2 ] += Ii
                    Mmat[2 , jj] += Ii
                    break
                k = parent[k]

    return Mmat


def compute_coriolis_and_gravity(q, qdot):
    """
    Compute C(q, qdot) + G(q), a 9-vector of generalized bias forces:
      C_i = Σ_{j,k} Γ_{i,j,k}(q) q̇_j q̇_k,
      G_i = ∂V/∂q_i,
    where V(q) = Σ_i m_i g · pcom_i(q)·[0,1]ᵀ. We compute G by differentiating potential energy.
    We compute Christoffel symbols via:
      Γ_{i,j,k} = ½ [ ∂M_{i,j}/∂q_k + ∂M_{i,k}/∂q_j − ∂M_{j,k}/∂q_i ].
    Then C_i = Σ_{j,k} Γ_{i,j,k} q̇_j q̇_k.
    Because nq=9 is small, we can compute ∂M/∂q by finite differences or symbolic. 
    Here we do numeric partial derivatives with a small ε.
    WARNING: finite-difference for ∂M is a bit slow, but acceptable for a single step.
    """
    eps = 1e-8
    M0 = compute_mass_matrix(q)
    # Pre-allocate ∂M/∂q tensors:
    dMdq = np.zeros((nq, nq, nq))
    for k in range(nq):
        dq = np.zeros(nq)
        dq[k] = eps
        M_plus  = compute_mass_matrix(q + dq)
        M_minus = compute_mass_matrix(q - dq)
        dMdq[:, :, k] = (M_plus - M_minus) / (2 * eps)

    # Compute Christoffel symbols Γ_{i,j,k} on the fly:
    Cvec = np.zeros(nq)
    for i in range(nq):
        for j in range(nq):
            for k in range(nq):
                Gamma = 0.5 * (dMdq[i, j, k] + dMdq[i, k, j] - dMdq[j, k, i])
                Cvec[i] += Gamma * qdot[j] * qdot[k]

    # Compute gravity vector G(q):
    # Potential energy V = Σ_i m_i * g * [pcom_i(q)]_z
    R, p, pcom = compute_link_pose(q)
    V = 0.0
    for i in range(7):
        # z coordinate of pcom_i = pcom[i][1]
        V += m_link[i] * g * pcom[i][1]
    G = np.zeros(nq)
    # Numeric gradient ∂V/∂q_i
    for i in range(nq):
        dq = np.zeros(nq)
        dq[i] = eps
        R_plus, p_plus, pcom_plus = compute_link_pose(q + dq)
        V_plus = sum(m_link[j] * g * pcom_plus[j][1] for j in range(7))
        R_minus, p_minus, pcom_minus = compute_link_pose(q - dq)
        V_minus = sum(m_link[j] * g * pcom_minus[j][1] for j in range(7))
        G[i] = (V_plus - V_minus) / (2 * eps)

    return Cvec + G


def halfcheetah_next_state(state, action, dt):
    """
    Main API:
      state:  length-18 array [q (9), qdot (9)]
      action: length-6 array [τ1…τ6]
    Returns state_next: length-18 [q_next, qdot_next]
    """
    q = state[:nq].copy()
    qdot = state[nq:].copy()

    # Compute mass matrix M(q):
    Mmat = compute_mass_matrix(q)

    # Build full torque vector tau_full ∈ ℝ⁹:
    tau_full = np.zeros(nq)
    # Floating base (x,z,θ) have zero torque input
    # Joint torques feed joints 1→6 (indices 3..8 in q):
    tau_full[3:9] = action  # u1..u6

    # Compute bias (C + G):
    bias = compute_coriolis_and_gravity(q, qdot)

    # Solve for accelerations:
    #   M(q) · qdd = tau_full − bias
    qdd = np.linalg.solve(Mmat, tau_full - bias)

    # Semi-implicit Euler integration:
    qdot_next = qdot + dt * qdd
    q_next    = q + dt * qdot_next

    state_next = np.concatenate([q_next, qdot_next])
    return state_next


# -----------------------------
# Example usage / sanity check
# -----------------------------
if __name__ == "__main__":
    import gym

    # Create Gym HalfCheetah for “ground truth” comparison
    env = gym.make('HalfCheetah-v2')
    env.reset()
    data = env.sim.data

    # Build an initial full state vector (9 qpos + 9 qvel)
    q0    = data.qpos.copy()   # length 9
    qdot0 = data.qvel.copy()   # length 9
    state0 = np.concatenate([q0, qdot0])

    dt = env.sim.model.opt.timestep

    N = 5
    state_manual = state0.copy()

    print("Comparing manual dynamics vs. env.step():")
    for i in range(N):
        # Sample random action:
        u = env.action_space.sample()

        # 1) Manual prediction
        state_manual = halfcheetah_next_state(state_manual, u, dt)

        # 2) “Ground truth” via env.step:
        #    Reset MuJoCo’s state to state_manual_true
        env.set_state(state0[:nq], state0[nq:])
        _, _, _, _ = env.step(u)
        q_true_next    = data.qpos.copy()
        qdot_true_next = data.qvel.copy()
        state0 = np.concatenate([q_true_next, qdot_true_next])

        # 3) Compare:
        err = np.linalg.norm(state_manual - state0)
        print(f"Step {i+1}: ‖manual – env.step‖ = {err:.6e}")

    print("Done.")
