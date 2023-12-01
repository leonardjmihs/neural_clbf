from typing import Tuple, Optional, List
from math import sqrt

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class LinearPoint(ControlAffineSystem):
    """
    Represents a pointmass with 

    The system has state

        x = [x, y, xdot, ydot]

    representing the position and velocity of the chaser satellite, and it
    has control inputs

        u = [ux, uy]

    representing the thrust applied in each axis. Distances are in km, and control
    inputs are measured in km/s^2.

    The task here is to get to the origin without leaving the bounding box [-5, 5] on
    all positions and [-1, 1] on velocities.

    The system is parameterized by
        a: the length of the semi-major axis of the target's orbit (e.g. 6871)
        ux_target, uy_target, uz_target: accelerations due to unmodelled effects and
                                         target control.
    """

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1
    XDOT = 2
    YDOT = 3
    # Control indices
    UX = 0
    UY = 1

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
        use_l1_norm: bool = False,
    ):
        """
        Initialize the inverted pendulum.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2
        raises:
            ValueError if nominal_params are not valid for this system
        """
        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )
        self.use_l1_norm = use_l1_norm

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise
        """
        valid = True

        return valid

    @property
    def n_dims(self) -> int:
        return LinearPoint.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return LinearPoint.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[LinearPoint.X] = 2.0
        upper_limit[LinearPoint.Y] = 2.0
        upper_limit[LinearPoint.XDOT] = 1
        upper_limit[LinearPoint.YDOT] = 1

        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.tensor([1.0, 1.0])
        lower_limit = -1.0 * upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        safe_mask = torch.ones_like(x[:, 0], dtype=torch.bool)

        # Stay within some maximum distance from the target
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearPoint.Y + 1].norm(dim=-1, p=order)
        # safe_mask.logical_and_(distance <= 1.0)

        # Stay at least some minimum distance from the target
        safe_mask.logical_and_(distance >= 0.75)

        return safe_mask

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of points in the state space
        """
        unsafe_mask = torch.zeros_like(x[:, 0], dtype=torch.bool)

        # Maximum distance
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : LinearPoint.Y + 1].norm(dim=-1, p=order)
        unsafe_mask.logical_or_(distance >= 1.5)

        # Minimum distance
        # unsafe_mask.logical_or_(distance <= 0.25)

        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of points in the state space
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        goal_mask = x[:, : LinearPoint.Y + 1].norm(dim=-1, p=order) <= 0.5

        return goal_mask

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1))
        f = f.type_as(x)

        xdot_ = x[:, LinearPoint.XDOT]
        ydot_ = x[:, LinearPoint.YDOT]

        # The first three dimensions just integrate the velocity
        f[:, LinearPoint.X, 0] = xdot_
        f[:, LinearPoint.Y, 0] = ydot_

        # The last three use the CHW equations
        f[:, LinearPoint.XDOT, 0] = 0
        f[:, LinearPoint.YDOT, 0] = 0

        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls))
        g = g.type_as(x)

        # The control inputs are accelerations
        g[:, LinearPoint.XDOT, LinearPoint.UX] = 1.0
        g[:, LinearPoint.YDOT, LinearPoint.UY] = 1.0

        return g
