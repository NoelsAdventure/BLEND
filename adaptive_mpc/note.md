Integrate adaptive_predictive_mpc_demo.py as a new evaluation policy/baseline in our CrowdNav/GenSafeNav code. Keep the MPC formulation, but replace the toy simulator pieces with the existing environment interfaces. Specifically:

Do not run another trajectory predictor. Replace predict_humans_constant_velocity() with the GST predictions already computed by our current observation/prediction pipeline.
Do not independently recompute awareness differently. Use exactly the same existing κt calculation used by adaptive_gt / adaptive so LoRA and Adaptive MPC receive identical κt
Set the MPC clearance every timestep as d_min = (1 - kappa_t) + d0.
Only provide MPC with pedestrians currently observable under the same sensor range/mask as the RL policy. Do not give MPC global pedestrian information.
Use the current environment timestep rather than assuming the toy simulator's DT.
GST predicted human positions should have shape conceptually [human, prediction_step, xy] and directly replace the constant-velocity predictions.
MPC outputs acceleration because Le et al. use a double-integrator model. Convert/integrate the first MPC acceleration into whatever action format the existing CrowdNav environment expects.
Execute only the first MPC action and re-solve at every environment timestep.
Add evaluation modes mpc_fixed and mpc_adaptive. mpc_fixed takes a configurable fixed d_min; mpc_adaptive uses (1-kappa)+d0.
Log at every timestep: kappa, d_min, MPC solve time, solver success/failure, robot action, number of sensed humans, and minimum human distance. Report the same SR/CR/NT/PL/ITR/SD metrics as the other methods.
set D0 = ROBOT_RADIUS + HUMAN_RADIUS