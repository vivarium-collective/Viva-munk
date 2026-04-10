"""Pymunk 2D physics process and placement helpers."""
import random
import math
import uuid

import pymunk

from process_bigraph import Process


def daughter_locations(parent_state, *, gap=1.0, daughter_length=None, daughter_radius=None):
    """
    Compute daughter center locations so they do not overlap, accounting for geometry.
    """
    px, py = parent_state['location']
    angle = float(parent_state.get('angle', 0.0))
    dtype = parent_state.get('type', 'circle')

    if dtype == 'segment':
        Lp = float(parent_state.get('length', 1.0))
        r  = float(parent_state.get('radius', 0.5))
        Ld = float(daughter_length) if daughter_length is not None else max(0.0, 0.5 * Lp)
        rd = float(daughter_radius) if daughter_radius is not None else r

        # Required center spacing to avoid overlap of two capsules
        d_center = Ld + 2.0 * rd + max(0.0, gap)
        hx = 0.5 * d_center * math.cos(angle)
        hy = 0.5 * d_center * math.sin(angle)

        return [[px - hx, py - hy],
                [px + hx, py + hy]]

    else:  # circle or fallback
        r  = float(parent_state.get('radius', 0.5))
        rd = float(daughter_radius) if daughter_radius is not None else r

        # Required center spacing to avoid overlap of two circles
        d_center = 2.0 * rd + max(0.0, gap)
        hx = 0.5 * d_center * math.cos(angle)
        hy = 0.5 * d_center * math.sin(angle)

        return [[px - hx, py - hy],
                [px + hx, py + hy]]

def local_impulse_point_for_shape(shape):
    """Return a random local point on the shape boundary in the body's local coords."""
    if isinstance(shape, pymunk.Circle):
        r = shape.radius
        theta = random.uniform(0, 2 * math.pi)
        return (r * math.cos(theta), r * math.sin(theta))

    if isinstance(shape, pymunk.Segment):
        # segment is from a->b in local coords; choose a random point along it, add small normal offset within radius
        t = random.random()
        ax, ay = shape.a
        bx, by = shape.b
        px, py = ax + t * (bx - ax), ay + t * (by - ay)
        # slight offset toward the “edge” to avoid perfectly central impulses
        nx, ny = -(by - ay), (bx - ax)  # unnormalized normal
        nlen = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nlen, ny / nlen
        return (px + nx * shape.radius, py + ny * shape.radius)

    # fallback: body center
    return (0.0, 0.0)


class PymunkProcess(Process):
    config_schema = {
        'env_size': {'_type': 'float', '_default': 500},
        # Optional rectangular chamber. If env_height is 0 or unset, the
        # chamber is square (height = env_size).
        'env_height': {'_type': 'float', '_default': 0.0},
        'substeps': {'_type': 'integer', '_default': 10},
        'damping_per_second': {'_type': 'float', '_default': 0.1},
        'gravity': {'_type': 'float', '_default': 0.0},
        'friction': {'_type': 'float', '_default': 0.8},
        'elasticity': {'_type': 'float', '_default': 0.0},
        'jitter_per_second': {'_type': 'float', '_default': 1e-4},  # (impulse std)
        # Bending cell options
        'n_bending_segments': {'_type': 'integer', '_default': 4},   # sub-segments per bending cell
        'bending_stiffness':  {'_type': 'float',   '_default': 100.0},  # rotary spring stiffness
        'bending_damping':    {'_type': 'float',   '_default': 5.0},   # rotary spring damping
        # Surface adhesion options
        'adhesion_enabled':    {'_type': 'boolean', '_default': False},
        'adhesion_surface':    {'_type': 'string',  '_default': 'bottom'},  # bottom/top/left/right
        'adhesion_threshold':  {'_type': 'float',   '_default': 0.5},  # min adhesins to attach
        'adhesion_distance':   {'_type': 'float',   '_default': 0.5},  # max distance to surface to attach
        'barriers': 'list[map]',
        'wall_thickness': {'_type': 'float', '_default': 5},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # create the environment
        self.space = pymunk.Space()
        self.space.gravity = (0, self.config['gravity'])  # Gravity set for the space
        self.substeps = int(self.config.get('substeps', 100))
        self.damping_per_second = float(self.config.get('damping_per_second', 0.98))
        self.jitter_per_second = float(self.config.get('jitter_per_second', 1e-2))

        self.agents = {}

        # add walls — chamber is env_size wide and env_height tall (or
        # env_size tall if env_height is 0 / unset, for backwards compat).
        env_size = self.config['env_size']
        env_height = float(self.config.get('env_height', 0.0) or 0.0)
        if env_height <= 0.0:
            env_height = float(env_size)
        self._env_width = float(env_size)
        self._env_height = env_height
        boundary_thickness = self.config.get('wall_thickness', 100)
        walls = [
            pymunk.Segment(self.space.static_body,
                           (-boundary_thickness, -boundary_thickness),
                           (env_size + boundary_thickness, -boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (env_size + boundary_thickness, -boundary_thickness),
                           (env_size + boundary_thickness, env_height + boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (env_size + boundary_thickness, env_height + boundary_thickness),
                           (-boundary_thickness, env_height + boundary_thickness),
                           boundary_thickness
                           ),
            pymunk.Segment(self.space.static_body,
                           (-boundary_thickness, env_height + boundary_thickness),
                           (-boundary_thickness, -boundary_thickness),
                           boundary_thickness
                           )
        ]
        for wall in walls:
            wall.elasticity = self.config['elasticity']
            wall.friction = self.config['friction']
            self.space.add(wall)

        # add custom barriers
        for barrier in self.config.get('barriers', []):
            self.add_barrier(barrier)

    def add_barrier(self, barrier):
        start_x, start_y = barrier['start']
        end_x, end_y = barrier['end']
        start = pymunk.Vec2d(start_x, start_y)
        end = pymunk.Vec2d(end_x, end_y)
        thickness = barrier.get('thickness', 1)
        segment = pymunk.Segment(self.space.static_body, start, end, thickness)
        segment.elasticity = barrier.get('elasticity', 1.0)
        segment.friction = barrier.get('friction', 0.5)
        self.space.add(segment)

    def inputs(self):
        return {
            'segment_cells':    'map[pymunk_agent]',  # rigid capsule cells (rod-shaped)
            'bending_cells':    'map[pymunk_agent]',  # multi-segment soft capsules with springs
            'circle_particles': 'map[pymunk_agent]',  # circular particles (e.g. EPS)
        }

    def outputs(self):
        return {
            'segment_cells':    'map[pymunk_agent]',
            'bending_cells':    'map[pymunk_agent]',
            'circle_particles': 'map[pymunk_agent]',
        }

    def update(self, inputs, interval):
        """
        Sync each port's bodies, step the physics, return per-port deltas.
        """
        segment_in = inputs.get('segment_cells', {}) or {}
        bending_in = inputs.get('bending_cells', {}) or {}
        circle_in  = inputs.get('circle_particles', {}) or {}

        # Sanity: IDs must be unique across ports this tick
        all_id_sets = [('segment_cells', set(segment_in)),
                       ('bending_cells', set(bending_in)),
                       ('circle_particles', set(circle_in))]
        for i, (n1, s1) in enumerate(all_id_sets):
            for n2, s2 in all_id_sets[i+1:]:
                overlap = s1 & s2
                if overlap:
                    raise ValueError(
                        f"PymunkProcess.update(): ID(s) present in both {n1} and {n2}: {sorted(overlap)}"
                    )

        segment_ids = set(segment_in)
        bending_ids = set(bending_in)
        circle_ids  = set(circle_in)

        # ------- sync bodies -------
        existing_ids = set(self.agents)
        new_ids = segment_ids | bending_ids | circle_ids

        # Remove stale agents (handle compound bodies for bending)
        for dead_id in existing_ids - new_ids:
            self._remove_object(dead_id)

        # Add/update each port's entries with the appropriate kind
        for _id, attrs in segment_in.items():
            self.manage_object(_id, attrs, kind='rigid')
        for _id, attrs in bending_in.items():
            self.manage_object(_id, attrs, kind='bending')
        for _id, attrs in circle_in.items():
            self.manage_object(_id, attrs, kind='rigid')

        # ------- integrate -------
        n_steps = max(1, int(self.config.get('substeps', 100)))
        dt = float(interval) / n_steps
        d_step = self.damping_per_second ** max(dt, 0.0)

        # Compute jitter sigma once (constant per substep)
        sigma = self.jitter_per_second * math.sqrt(max(dt, 1e-12))
        # Skip jitter entirely if sigma is below numerical noise floor
        apply_jitter = sigma > 1e-12

        adhesion_enabled = bool(self.config.get('adhesion_enabled', False))
        # Particles get a fraction of the cell jitter (so they drift but don't jitter wildly)
        particle_jitter_fraction = float(self.config.get('particle_jitter_fraction', 0.2))
        sigma_particle = sigma * particle_jitter_fraction
        particle_body_ids = set()
        if apply_jitter:
            for obj in self.agents.values():
                if obj.get('type') == 'circle':
                    particle_body_ids.add(id(obj['body']))
        # Pre-collect cells driven by chemotaxis (or any caller that injects
        # thrust/torque/motile_speed). We key off the *presence* of the
        # motile_speed key — chemotaxis cells in tumble state still need
        # velocity hard-zeroed each substep — but non-chemotaxis cells must
        # NOT carry that key, otherwise the substep loop would zero their
        # velocity every step and prevent collisions from pushing them
        # (this broke mother machine until create_new_object stopped
        # auto-inserting motile_speed=0.0).
        motile_cells = [
            (a['body'], a)
            for a in self.agents.values()
            if a.get('kind') == 'rigid'
            and a.get('type') == 'segment'
            and (
                'motile_speed' in a
                or a.get('thrust', 0.0)
                or a.get('torque', 0.0)
            )
        ]
        for _ in range(n_steps):
            self.space.damping = d_step
            if apply_jitter:
                for body in self.space.bodies:
                    if id(body) in particle_body_ids:
                        if sigma_particle > 1e-12:
                            self.apply_jitter_force(body, dt, sigma_particle)
                    else:
                        self.apply_jitter_force(body, dt, sigma)
            for body, agent in motile_cells:
                # Direct velocity control for chemotaxis cells: when
                # `motile_speed` is set the chemotaxis process owns the
                # cell's swimming speed, and we override body.velocity each
                # substep so the cell really moves at exactly that speed
                # along its current heading. Setting motile_speed = 0
                # gives a hard stop (used during tumbles).
                if 'motile_speed' in agent:
                    speed = float(agent.get('motile_speed', 0.0) or 0.0)
                    ang = body.angle
                    body.velocity = (speed * math.cos(ang), speed * math.sin(ang))
                # Force-based thrust / torque (still supported alongside
                # the velocity-control path).
                thrust = float(agent.get('thrust', 0.0) or 0.0)
                torque = float(agent.get('torque', 0.0) or 0.0)
                if thrust != 0.0:
                    body.apply_force_at_local_point((thrust, 0.0), (0.0, 0.0))
                if torque != 0.0:
                    body.torque += torque
            self.space.step(dt)
        # Run adhesion once per process tick (not per substep) — cheap and sufficient
        if adhesion_enabled:
            self._apply_adhesion()

        # ------- emit deltas per port -------
        all_inputs = {}
        all_inputs.update(segment_in)
        all_inputs.update(bending_in)
        all_inputs.update(circle_in)
        segment_out = {}
        bending_out = {}
        circle_out  = {}

        for _id, obj in self.agents.items():
            if _id not in all_inputs:
                continue
            old = all_inputs[_id]
            old_loc = old.get('location', (0.0, 0.0))
            old_vel = old.get('velocity', (0.0, 0.0))
            old_angle = float(old.get('angle', 0.0) or 0.0)
            old_inertia = float(old.get('inertia', 0.0) or 0.0)

            # Get aggregate position/velocity (uses _aggregate for compound bodies)
            new_loc, new_vel, new_angle, new_inertia = self._aggregate(obj)

            rec = {
                'type': obj['type'],  # Enum has set semantics — must always be returned
                'location': (new_loc[0] - old_loc[0], new_loc[1] - old_loc[1]),
                'velocity': (new_vel[0] - old_vel[0], new_vel[1] - old_vel[1]),
                'angle': new_angle - old_angle,
                'inertia': new_inertia - old_inertia,
            }

            if _id in segment_ids:
                # Expose the attached state so other processes (e.g. GrowDivide) can read it
                rec['attached'] = 1.0 if obj.get('attached') else 0.0
                segment_out[_id] = rec
            elif _id in bending_ids:
                # Include the actual bent spine for visualization
                rec['polyline'] = self._bending_polyline(obj)
                bending_out[_id] = rec
            elif _id in circle_ids:
                circle_out[_id] = rec

        return {
            'segment_cells':    segment_out,
            'bending_cells':    bending_out,
            'circle_particles': circle_out,
        }

    def _bending_polyline(self, obj):
        """Return the bent spine of a bending cell as a list of (x, y) world points.

        For N sub-segments, returns N+1 points: start of segment 0, then the
        shared endpoint between each consecutive pair, then end of segment N-1.
        """
        bodies = obj['bodies']
        n = len(bodies)
        if n == 0:
            return []
        # Each sub-segment is from (-half_len, 0) to (+half_len, 0) in body-local coords
        half_len = obj['length'] / (2 * n)
        points = []
        # Start of first segment in world coords
        first = bodies[0]
        points.append(first.local_to_world((-half_len, 0)))
        # End of each segment
        for body in bodies:
            p = body.local_to_world((half_len, 0))
            points.append((p.x, p.y))
        # Convert any Vec2d to plain tuples for serialization
        return [(p[0], p[1]) if not hasattr(p, 'x') else (p.x, p.y) for p in points]

    def _aggregate(self, obj):
        """Return (location, velocity, angle, inertia) for either rigid or compound body."""
        if obj.get('kind') == 'bending':
            bodies = obj['bodies']
            n = len(bodies)
            cx = sum(b.position.x for b in bodies) / n
            cy = sum(b.position.y for b in bodies) / n
            vx = sum(b.velocity.x for b in bodies) / n
            vy = sum(b.velocity.y for b in bodies) / n
            # Average angle is fine since rest_angles are 0 — use end-to-end direction
            first, last = bodies[0], bodies[-1]
            ang = math.atan2(last.position.y - first.position.y,
                             last.position.x - first.position.x)
            inertia = sum(b.moment for b in bodies)
            return (cx, cy), (vx, vy), ang, inertia
        else:
            body = obj['body']
            ang = body.angle if obj['type'] == 'segment' else 0.0
            return ((body.position.x, body.position.y),
                    (body.velocity.x, body.velocity.y),
                    ang, body.moment)

    def _remove_object(self, agent_id):
        obj = self.agents[agent_id]
        # Remove adhesion joints if present
        for key in ('attachment_joint', 'cluster_joint'):
            joint = obj.get(key)
            if joint is not None:
                try:
                    self.space.remove(joint)
                except Exception:
                    pass
        if obj.get('kind') == 'bending':
            for joint in obj.get('joints', []):
                self.space.remove(joint)
            for shape in obj['shapes']:
                self.space.remove(shape)
            for body in obj['bodies']:
                self.space.remove(body)
        else:
            self.space.remove(obj['body'], obj['shape'])
        del self.agents[agent_id]

    def apply_jitter_force(self, body, dt, sigma):
        # Apply jitter at body center (skip per-shape lookup for speed)
        fx = random.gauss(0.0, sigma)
        fy = random.gauss(0.0, sigma)
        body.apply_impulse_at_local_point((fx, fy), (0.0, 0.0))

    def _apply_adhesion(self):
        """Pin eligible bodies to the adhesion surface or to other attached bodies.

        Cells (segments) attach when they touch the surface and have enough adhesins.
        Particles (circles) attach when they touch the surface OR another already-attached body.
        Once attached, a PivotJoint anchors the body at its current position.
        """
        surface = self.config.get('adhesion_surface', 'bottom')
        threshold = float(self.config.get('adhesion_threshold', 0.5))
        max_dist = float(self.config.get('adhesion_distance', 0.5))
        env_size = float(self.config['env_size'])

        def _surface_distance(x, y, r_along_normal):
            """Distance from outermost edge to the configured surface (positive = away)."""
            if surface == 'bottom':
                return y - r_along_normal
            elif surface == 'top':
                return (env_size - y) - r_along_normal
            elif surface == 'left':
                return x - r_along_normal
            elif surface == 'right':
                return (env_size - x) - r_along_normal
            return float('inf')

        # First pass: cells touching the surface (with adhesins) get attached
        for agent in self.agents.values():
            if agent.get('attached'):
                continue
            if agent.get('kind') != 'rigid' or agent.get('type') != 'segment':
                continue
            if agent.get('adhesins', 0.0) < threshold:
                continue
            body = agent['body']
            r = agent['radius']
            L = agent.get('length') or 0.0
            cx, cy = body.position.x, body.position.y
            ang = body.angle
            half = L / 2.0
            ex = half * math.cos(ang)
            ey = half * math.sin(ang)
            # Distance from outermost capsule edge to the surface
            if surface in ('bottom', 'top'):
                outer = min(cy - ey, cy + ey) if surface == 'bottom' else max(cy - ey, cy + ey)
                d = _surface_distance(cx, outer, r)
            else:
                outer = min(cx - ex, cx + ex) if surface == 'left' else max(cx - ex, cx + ex)
                d = _surface_distance(outer, cy, r)
            if d > max_dist:
                continue
            joint = pymunk.PivotJoint(
                self.space.static_body, body, body.position, (0, 0))
            self.space.add(joint)
            agent['attached'] = True
            agent['attachment_joint'] = joint

        # Second pass: circle particles
        circle_agents = [a for a in self.agents.values()
                         if a.get('kind') == 'rigid' and a.get('type') == 'circle']
        for agent in circle_agents:
            if agent.get('attached'):
                continue
            body = agent['body']
            r = agent['radius']
            cx, cy = body.position.x, body.position.y

            # 1) Direct contact with the wall — pin to static body
            d = _surface_distance(cx, cy, r)
            if d <= max_dist:
                joint = pymunk.PivotJoint(
                    self.space.static_body, body, body.position, (0, 0))
                self.space.add(joint)
                agent['attached'] = True
                agent['attachment_joint'] = joint
                continue

            # 2) Contact with another circle — link them so they move as a unit.
            #    Each particle keeps at most one cluster joint to avoid over-constraint.
            if agent.get('cluster_joint'):
                continue
            for other in circle_agents:
                if other is agent:
                    continue
                ob = other['body']
                ox, oy = ob.position.x, ob.position.y
                or_ = other['radius']
                contact = r + or_ + max_dist
                dx, dy = cx - ox, cy - oy
                if dx * dx + dy * dy <= contact * contact:
                    # Anchor at the midpoint between the two particles, in each body's local frame
                    mx = (cx + ox) / 2.0
                    my = (cy + oy) / 2.0
                    local_a = (mx - cx, my - cy)
                    local_b = (mx - ox, my - oy)
                    joint = pymunk.PivotJoint(body, ob, local_a, local_b)
                    self.space.add(joint)
                    agent['cluster_joint'] = joint
                    break

    def update_bodies(self, agents):
        existing_ids = set(self.agents.keys())
        new_ids = set(agents.keys())

        # Remove objects not in the new state
        for agent_id in existing_ids - new_ids:
            body = self.agents[agent_id]['body']
            shape = self.agents[agent_id]['shape']
            self.space.remove(body, shape)
            del self.agents[agent_id]

        # Add or update existing objects
        for agent_id, attrs in agents.items():
            self.manage_object(agent_id, attrs)

    def manage_object(self, agent_id, attrs, kind='rigid'):
        agent = self.agents.get(agent_id)
        if not agent:
            self.create_new_object(agent_id, attrs, kind=kind)
            return

        # If kind changed (shouldn't normally), rebuild from scratch
        if agent.get('kind', 'rigid') != kind:
            self._remove_object(agent_id)
            self.create_new_object(agent_id, attrs, kind=kind)
            return

        if kind == 'bending':
            self._update_bending(agent_id, agent, attrs)
            return

        body = agent['body']
        old_shape = agent['shape']
        old_type = agent['type']

        # robust defaults — type may be None if not in the latest update (Enum has set semantics)
        shape_type = attrs.get('type') or old_type or 'circle'
        mass = float(attrs.get('mass', body.mass))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        body.mass = mass
        body.position = pymunk.Vec2d(*attrs.get('location', (body.position.x, body.position.y)))
        body.velocity = pymunk.Vec2d(vx, vy)

        needs_rebuild = False
        if shape_type == 'circle':
            radius = float(attrs['radius'])
            if not isinstance(old_shape, pymunk.Circle) or abs(
                    old_shape.radius - radius) > 1e-9 or old_type != 'circle':
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Circle(body, radius)
                body.moment = pymunk.moment_for_circle(mass, 0, radius)
        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])
            # Local endpoints along the x-axis — body.angle handles world rotation
            start = pymunk.Vec2d(-length / 2, 0)
            end = pymunk.Vec2d(length / 2, 0)
            # Rebuild whenever length, radius, or type changes
            old_length = agent.get('length') or 0.0
            if (not isinstance(old_shape, pymunk.Segment)
                    or abs(old_shape.radius - radius) > 1e-9
                    or abs(old_length - length) > 1e-9
                    or old_type != 'segment'):
                needs_rebuild = True
            if needs_rebuild:
                new_shape = pymunk.Segment(body, start, end, radius)
                body.moment = pymunk.moment_for_segment(mass, start, end, radius)
            body.angle = angle
            body.length = length
            body.width = radius * 2
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        # swap shape if needed
        if needs_rebuild:
            # preserve material params
            elasticity = attrs.get('elasticity', self.config['elasticity'])
            friction = attrs.get('friction', self.config['friction'])
            new_shape.elasticity = elasticity
            new_shape.friction = friction

            self.space.remove(old_shape)
            self.space.add(new_shape)
            agent['shape'] = new_shape

        # keep dict in sync
        agent['type'] = shape_type
        agent['mass'] = mass
        if shape_type == 'circle':
            agent['radius'] = radius
            agent['angle'] = None
            agent['length'] = None
        else:
            agent['radius'] = radius
            agent['angle'] = body.angle
            agent['length'] = body.length
        # Track adhesin count if provided (used by adhesion logic)
        if 'adhesins' in attrs and attrs['adhesins'] is not None:
            agent['adhesins'] = float(attrs['adhesins'])
        # Sync chemotaxis thrust/torque from the framework state into the
        # internal agent dict so the substep loop can apply them as forces.
        agent['thrust'] = float(attrs.get('thrust', 0.0) or 0.0)
        agent['torque'] = float(attrs.get('torque', 0.0) or 0.0)
        # Sync chemotaxis motile_speed (cell swimming speed in m/s). The
        # substep loop reads this and sets body.velocity directly each
        # iteration so the cell moves at exactly this speed in its
        # current heading direction (or stops at 0 during a tumble).
        if 'motile_speed' in attrs and attrs['motile_speed'] is not None:
            agent['motile_speed'] = float(attrs['motile_speed'])

    def create_new_object(self, agent_id, attrs, kind='rigid'):
        if kind == 'bending':
            self._create_bending(agent_id, attrs)
            return

        shape_type = attrs.get('type') or 'circle'
        mass = float(attrs.get('mass', 1.0))
        vx, vy = attrs.get('velocity', (0.0, 0.0))
        pos = attrs.get('location', (0.0, 0.0))

        if shape_type == 'circle':
            radius = float(attrs['radius'])
            inertia = pymunk.moment_for_circle(mass, 0, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            shape = pymunk.Circle(body, radius)
            angle = None
            length = None
        elif shape_type == 'segment':
            length = float(attrs['length'])
            radius = float(attrs['radius'])
            angle = float(attrs['angle'])
            start = (-length / 2, 0)
            end = (length / 2, 0)
            inertia = pymunk.moment_for_segment(mass, start, end, radius)
            body = pymunk.Body(mass, inertia)
            body.position = pos
            body.velocity = (vx, vy)
            body.angle = angle
            body.length = length
            body.width = radius * 2
            shape = pymunk.Segment(body, start, end, radius)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        shape.elasticity = attrs.get('elasticity', self.config['elasticity'])
        shape.friction = attrs.get('friction', self.config['friction'])

        self.space.add(body, shape)
        self.agents[agent_id] = {
            'kind': 'rigid',
            'body': body,
            'shape': shape,
            'type': shape_type,
            'mass': mass,
            'radius': radius,
            'angle': angle,
            'length': length,
            'adhesins': float(attrs.get('adhesins', 0.0) or 0.0),
            'attached': False,
            'attachment_joint': None,
            'thrust': float(attrs.get('thrust', 0.0) or 0.0),
            'torque': float(attrs.get('torque', 0.0) or 0.0),
        }
        # Only attach motile_speed if the caller explicitly set it (e.g.
        # a chemotaxis cell that has the key in its initial state).
        # Otherwise leave it absent so the substep loop's
        # `'motile_speed' in agent` test stays False for normal cells and
        # we don't end up zeroing their velocities every substep.
        if 'motile_speed' in attrs and attrs['motile_speed'] is not None:
            self.agents[agent_id]['motile_speed'] = float(attrs['motile_speed'])

    # ------------------------------------------------------------------
    # Bending cell support — multi-segment compound bodies
    # ------------------------------------------------------------------

    def _create_bending(self, agent_id, attrs):
        """Build a bending cell as N rigid sub-segments linked by pivots + rotary springs."""
        n = int(self.config.get('n_bending_segments', 4))
        stiffness = float(self.config.get('bending_stiffness', 100.0))
        damping = float(self.config.get('bending_damping', 5.0))

        total_mass = float(attrs.get('mass', 1.0))
        total_length = float(attrs['length'])
        radius = float(attrs['radius'])
        angle = float(attrs.get('angle', 0.0))
        cx, cy = attrs.get('location', (0.0, 0.0))
        vx, vy = attrs.get('velocity', (0.0, 0.0))

        sub_len = total_length / n
        sub_mass = total_mass / n
        sub_inertia = pymunk.moment_for_segment(
            sub_mass, (-sub_len / 2, 0), (sub_len / 2, 0), radius)
        elasticity = attrs.get('elasticity', self.config['elasticity'])
        friction = attrs.get('friction', self.config['friction'])

        bodies, shapes, joints = [], [], []
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Center positions of each sub-segment, spaced along the cell axis
        for i in range(n):
            t = (i - (n - 1) / 2.0) * sub_len  # offset from cell center
            bx = cx + cos_a * t
            by = cy + sin_a * t
            body = pymunk.Body(sub_mass, sub_inertia)
            body.position = (bx, by)
            body.velocity = (vx, vy)
            body.angle = angle
            shape = pymunk.Segment(body, (-sub_len / 2, 0), (sub_len / 2, 0), radius)
            shape.elasticity = elasticity
            shape.friction = friction
            self.space.add(body, shape)
            bodies.append(body)
            shapes.append(shape)

        # Disable collisions BETWEEN sub-segments of the same cell using a unique group
        group_id = id(bodies[0]) & 0x7FFFFFFF or 1
        sf = pymunk.ShapeFilter(group=group_id)
        for s in shapes:
            s.filter = sf

        # Pin adjacent sub-bodies at their shared endpoint and add a rotary spring
        for i in range(n - 1):
            a, b = bodies[i], bodies[i + 1]
            # Anchor at the right end of `a` (= left end of `b`) in each body's local frame
            pivot = pymunk.PivotJoint(a, b, (sub_len / 2, 0), (-sub_len / 2, 0))
            spring = pymunk.DampedRotarySpring(
                a, b, rest_angle=0.0, stiffness=stiffness, damping=damping)
            self.space.add(pivot, spring)
            joints.append(pivot)
            joints.append(spring)

        self.agents[agent_id] = {
            'kind': 'bending',
            'type': 'segment',
            'bodies': bodies,
            'shapes': shapes,
            'joints': joints,
            'mass': total_mass,
            'radius': radius,
            'length': total_length,
            'angle': angle,
            'n_segments': n,
        }

    def _update_bending(self, agent_id, agent, attrs):
        """Update an existing bending cell. Preserves the current bent shape
        when growing — old sub-body positions are scaled outward from the
        centroid so the cell appears to grow continuously.
        """
        new_length = float(attrs.get('length', agent['length']) or agent['length'])
        new_radius = float(attrs.get('radius', agent['radius']) or agent['radius'])
        new_mass = float(attrs.get('mass', agent['mass']) or agent['mass'])

        old_length = agent['length']
        old_radius = agent['radius']

        if (abs(new_length - old_length) > 1e-6
                or abs(new_radius - old_radius) > 1e-6):
            # Snapshot current sub-body positions, velocities, angles BEFORE removing
            old_bodies = agent['bodies']
            n = len(old_bodies)
            cx = sum(b.position.x for b in old_bodies) / n
            cy = sum(b.position.y for b in old_bodies) / n
            old_positions = [(b.position.x - cx, b.position.y - cy) for b in old_bodies]
            old_velocities = [(b.velocity.x, b.velocity.y) for b in old_bodies]
            old_angles = [b.angle for b in old_bodies]

            # Scale offsets outward from centroid in proportion to length growth
            scale = new_length / old_length if old_length > 0 else 1.0

            # Remove the old compound and rebuild at the same scaled-out positions
            self._remove_object(agent_id)

            n_new = int(self.config.get('n_bending_segments', 4))
            if n_new != n:
                n_new = n  # keep same n to reuse the saved positions
            stiffness = float(self.config.get('bending_stiffness', 100.0))
            damping = float(self.config.get('bending_damping', 5.0))
            elasticity = attrs.get('elasticity', self.config['elasticity'])
            friction = attrs.get('friction', self.config['friction'])

            sub_len = new_length / n_new
            sub_mass = new_mass / n_new
            sub_inertia = pymunk.moment_for_segment(
                sub_mass, (-sub_len / 2, 0), (sub_len / 2, 0), new_radius)

            bodies, shapes, joints = [], [], []
            for i in range(n_new):
                ox, oy = old_positions[i]
                body = pymunk.Body(sub_mass, sub_inertia)
                body.position = (cx + ox * scale, cy + oy * scale)
                body.velocity = old_velocities[i]
                body.angle = old_angles[i]
                shape = pymunk.Segment(
                    body, (-sub_len / 2, 0), (sub_len / 2, 0), new_radius)
                shape.elasticity = elasticity
                shape.friction = friction
                self.space.add(body, shape)
                bodies.append(body)
                shapes.append(shape)

            # Disable collisions among sub-segments of this cell
            group_id = id(bodies[0]) & 0x7FFFFFFF or 1
            sf = pymunk.ShapeFilter(group=group_id)
            for s in shapes:
                s.filter = sf

            # Pin adjacent sub-bodies and add rotary springs
            for i in range(n_new - 1):
                a, b = bodies[i], bodies[i + 1]
                pivot = pymunk.PivotJoint(
                    a, b, (sub_len / 2, 0), (-sub_len / 2, 0))
                spring = pymunk.DampedRotarySpring(
                    a, b, rest_angle=0.0, stiffness=stiffness, damping=damping)
                self.space.add(pivot, spring)
                joints.append(pivot)
                joints.append(spring)

            self.agents[agent_id] = {
                'kind': 'bending',
                'type': 'segment',
                'bodies': bodies,
                'shapes': shapes,
                'joints': joints,
                'mass': new_mass,
                'radius': new_radius,
                'length': new_length,
                'angle': agent.get('angle', 0.0),
                'n_segments': n_new,
                'adhesins': agent.get('adhesins', 0.0),
            }
            return

        # Mass-only change: redistribute across sub-bodies
        if abs(new_mass - agent['mass']) > 1e-9:
            sub_mass = new_mass / agent['n_segments']
            for body in agent['bodies']:
                body.mass = sub_mass
            agent['mass'] = new_mass

    def get_state_update(self):
        state = {}
        for agent_id, obj in self.agents.items():
            if obj['type'] == 'circle':

                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'inertia': obj['body'].moment,
                }
            elif obj['type'] == 'segment':
                state[agent_id] = {
                    'type': obj['type'],
                    'location': (obj['body'].position.x, obj['body'].position.y),
                    'velocity': (obj['body'].velocity.x, obj['body'].velocity.y),
                    'inertia': obj['body'].moment,
                    'angle': obj['body'].angle
                }
        return state


# -------------------------
# Utilities
# -------------------------

def make_id(prefix='id', nhex=6):
    return f"{prefix}_{uuid.uuid4().hex[:nhex]}"

def make_rng(seed=None):
    return random.Random(seed)

# -------------------------
# Mass/geometry conversions
# -------------------------

def circle_mass_from_radius(radius, density):
    # m = ρ π r^2
    return float(density) * math.pi * (float(radius) ** 2)

def circle_radius_from_mass(mass, density):
    # r = sqrt(m / (ρ π))
    return math.sqrt(float(mass) / (float(density) * math.pi))

def capsule_mass_from_length_radius(length, radius, density):
    # Approximate capsule as rectangle length L and diameter 2r:
    # m = ρ * (2r) * L  (ignoring hemispherical ends for simplicity; good for L >> r)
    return float(density) * (2.0 * float(radius)) * float(length)

def capsule_length_from_mass(mass, radius, density):
    # L = m / (ρ * 2r)
    return float(mass) / (float(density) * (2.0 * float(radius)))

# -------------------------
# Primitive builders (single objects)
# - Accept explicit geometry/mass; if one is missing, infer from density.
# - Velocity can be set directly, or drawn from (speed_range, heading).
# -------------------------

def build_particle(
    rng,
    env_size,
    *,
    elasticity=0.0,
    id_prefix='p',
    # position
    x=None, y=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 10.0),
    # geometry / mass (circle)
    radius=None, mass=None, density=0.015,
    radius_range=None,
    radius_dist='uniform',  # 'uniform' or 'log_uniform' (scale-free)
):
    # derive geometry/mass if needed
    if radius is None and mass is None:
        rng_lo, rng_hi = radius_range if radius_range else (1.0, 10.0)
        if radius_dist == 'log_uniform' and rng_lo > 0 and rng_hi > rng_lo:
            # Sample uniformly in log-space → equal density per decade of size,
            # so the resulting distribution is approximately scale-free.
            radius = math.exp(rng.uniform(math.log(rng_lo), math.log(rng_hi)))
        else:
            radius = rng.uniform(rng_lo, rng_hi)
    if mass is None:
        mass = circle_mass_from_radius(radius, density)
    if radius is None:
        radius = circle_radius_from_mass(mass, density)

    # position
    if x is None or y is None:
        r = radius
        x = rng.uniform(margin + r, env_size - (margin + r))
        y = rng.uniform(margin + r, env_size - (margin + r))

    # velocity
    if velocity is None:
        speed = rng.uniform(*speed_range)
        theta = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(theta), speed * math.sin(theta)
    else:
        vx, vy = velocity

    return make_id(id_prefix), {
        'type': 'circle',
        'mass': float(mass),
        'radius': float(radius),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
    }

def build_microbe(
    rng,
    env_size,
    *,
    agent_id=None,          # <-- optional explicit id
    elasticity=0.0,
    id_prefix='a',
    # placement & bounds
    x=None, y=None, angle=None, margin=5.0,
    # kinematics
    velocity=None, speed_range=(0.0, 0.4),
    # geometry / mass (capsule segment)
    length=None, radius=None, mass=None, density=0.02,
    length_range=(40.0, 120.0), radius_range=(6.0, 24.0),
    adhesins=1.0,
):
    if length is None and mass is None:
        length = rng.uniform(*length_range)
    if radius is None:
        radius = rng.uniform(*radius_range)

    if mass is None:
        mass = capsule_mass_from_length_radius(length, radius, density)
    if length is None:
        length = capsule_length_from_mass(mass, radius, density)

    if angle is None:
        angle = rng.uniform(-math.pi, math.pi)

    dx, dy = (length / 2.0) * math.cos(angle), (length / 2.0) * math.sin(angle)
    pad = radius + margin
    if x is None or y is None:
        x = rng.uniform(pad + abs(dx), env_size - (pad + abs(dx)))
        y = rng.uniform(pad + abs(dy), env_size - (pad + abs(dy)))

    if velocity is None:
        speed = rng.uniform(*speed_range)
        phi = rng.uniform(0, 2 * math.pi)
        vx, vy = speed * math.cos(phi), speed * math.sin(phi)
    else:
        vx, vy = velocity

    _id = agent_id or make_id(id_prefix)
    return _id, {
        'id': _id,                 # <-- include id in the object
        'type': 'segment',
        'mass': float(mass),
        'length': float(length),
        'radius': float(radius),
        'angle': float(angle),
        'location': (float(x), float(y)),
        'velocity': (float(vx), float(vy)),
        'elasticity': float(elasticity),
        'adhesins': float(adhesins),
    }

# -------------------------
# Placers (collections)
# -------------------------

def circles_overlap(c1, c2, extra_gap=0.0):
    (x1, y1), r1 = c1['location'], c1['radius']
    (x2, y2), r2 = c2['location'], c2['radius']
    dx, dy = x1 - x2, y1 - y2
    return (dx*dx + dy*dy) < (r1 + r2 + extra_gap) ** 2

def place_circles(
    rng, env_size, n,
    *,
    margin=5.0,
    avoid_overlap=True,
    extra_gap=2.0,
    max_tries=200,
    particle_kwargs=None
):
    particle_kwargs = dict(particle_kwargs or {})
    placed = []
    out = {}
    for _ in range(n):
        if avoid_overlap:
            for _try in range(max_tries):
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                if all(not circles_overlap(cand, prev, extra_gap) for prev in placed):
                    placed.append(cand)
                    out[pid] = cand
                    break
            else:
                pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
                placed.append(cand)
                out[pid] = cand
        else:
            pid, cand = build_particle(rng, env_size, margin=margin, **particle_kwargs)
            placed.append(cand)
            out[pid] = cand
    return out

def place_microbes(
    rng, env_size, n,
    *,
    margin=5.0,
    microbe_kwargs=None,
    ids=None,                 # <-- optional list of ids (len == n)
    id_factory=None,          # <-- optional callable -> str
):
    microbe_kwargs = dict(microbe_kwargs or {})
    out = {}
    for i in range(n):
        agent_id = None
        if ids is not None:
            agent_id = ids[i]
        elif id_factory is not None:
            agent_id = id_factory(i)
        aid, obj = build_microbe(rng, env_size, margin=margin, agent_id=agent_id, **microbe_kwargs)
        out[aid] = obj
    return out


# -------------------------
# High-level initializer
# -------------------------

def make_initial_state(
    n_microbes=2,
    n_particles=2,
    env_size=600.0,
    *,
    agents_key='cells',
    particles_key='particles',
    seed=None,
    elasticity=0.0,
    # particle defaults
    particle_radius_range=(1.0, 10.0),
    particle_radius_dist='uniform',  # 'uniform' or 'log_uniform'
    particle_mass_density=0.015,
    particle_speed_range=(0.0, 10.0),
    # microbe defaults
    microbe_length_range=(40.0, 120.0),
    microbe_radius_range=(6.0, 24.0),
    microbe_mass_density=0.02,
    microbe_speed_range=(0.0, 0.4),
    # placement
    margin=5.0,
    avoid_overlap_circles=True,
    min_gap=2.0,
    max_tries_per_circle=200,
):
    rng = make_rng(seed)

    particles = place_circles(
        rng, env_size, n_particles,
        margin=margin,
        avoid_overlap=avoid_overlap_circles,
        extra_gap=min_gap,
        max_tries=max_tries_per_circle,
        particle_kwargs=dict(
            elasticity=elasticity,
            density=particle_mass_density,
            speed_range=particle_speed_range,
            radius_range=particle_radius_range,
            radius_dist=particle_radius_dist,
        )
    )

    agents = place_microbes(
        rng, env_size, n_microbes,
        margin=margin,
        microbe_kwargs=dict(
            elasticity=elasticity,
            density=microbe_mass_density,
            speed_range=microbe_speed_range,
            length_range=microbe_length_range,
            radius_range=microbe_radius_range,
        ),
        # e.g., fixed IDs:
        # ids=[f"a_seed{i}" for i in range(n_microbes)],
        # or dynamic:
        # id_factory=lambda i: f"a_{uuid.uuid4().hex[:6]}",
    )

    return {agents_key: agents, particles_key: particles}



def get_mother_machine_config(
    env_size=600,
    spacer_thickness=5,
    channel_height=500,
    channel_space=50
):
    barriers = []
    y_start = 0  # Start at the bottom of the environment
    y_end = channel_height  # Height of the channel

    # Calculate how many barriers can fit within the env_size
    num_channels = int(env_size / (spacer_thickness + channel_space))

    # Generate barriers based on calculated number
    x_position = spacer_thickness + channel_space
    for _ in range(num_channels):
        barrier = {
            'start': (x_position, y_start),
            'end': (x_position, y_end),
            'thickness': spacer_thickness
        }
        barriers.append(barrier)

        # Update x_position for the next barrier, adding the space for the channel
        x_position += spacer_thickness + channel_space

    return {
        'env_size': env_size,
        'barriers': barriers
    }
