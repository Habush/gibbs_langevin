import jax.random as random
from sgmcmc import *
import operator
import numpy as np

def get_discrete_kernel(seed, step_size_fn, log_prob_fn,
                        x0, mh=False,
                        temp=1.0, preconditioner=None,
                        cat=False, dim=None, num_cls=None):

    if cat:
        sampler = disc_cat_sgld_gradient_update(step_size_fn, seed, dim, num_cls,
                               preconditioner=preconditioner, mh=mh, temp=temp)
    else:
        sampler = disc_bin_sgld_gradient_update(step_size_fn, seed,
                               preconditioner=preconditioner, mh=mh, temp=temp)
    opt_state = sampler.init(x0)

    def step(z, state, opt_state):
        def lp_fn(x):
            return log_prob_fn(x, state)

        z, opt_state, accept_prob = sampler.update(z, lp_fn, opt_state)
        return z, opt_state, {"accept_prob": accept_prob}

    return step, opt_state

def get_continuous_kernel(seed, step_size_fn, log_prob_fn,
                          x0, mh=False,
                          preconditioner=get_rmsprop_preconditioner(), momentum=0.9):

    sampler = sgld_gradient_update(step_size_fn, seed,
                                   preconditioner=preconditioner,
                                   momentum_decay=momentum)
    init_opt_state = sampler.init(x0)

    def proposal_dist(x_new, x_prev, state, step_size):
        grad = jax.grad(log_prob_fn)(x_prev, state)
        theta = x_new - x_prev - step_size*grad
        # theta_dot = jnp.linalg.norm(theta)**2
        theta_dot = jax.tree_util.tree_reduce(
            operator.add, jax.tree_util.tree_map(lambda x: jnp.sum(x * x), theta))

        return -0.25*(1.0 / step_size) * theta_dot

    def step(x, state, opt_state):
        grad = jax.grad(log_prob_fn)(x, state)
        updates, opt_state = sampler.update(grad, opt_state)
        x_new = optax.apply_updates(x, updates)
        accept_prob = 1.0 #default in the unadjusted case where mh=False
        accepted = True
        if mh:
            step_size = step_size_fn(opt_state.count - 1) #minus 1 b/c the count has been incremented in the update
            q_forward = proposal_dist(x_new, x, state, step_size)
            log_prob_x = log_prob_fn(x, state)
            q_reverse = proposal_dist(x, x_new, state, step_size)
            log_prob_x_new = log_prob_fn(x_new, state)

            m = (log_prob_x_new - log_prob_x) + (q_reverse - q_forward)
            delta = jnp.exp(m)
            delta = jnp.where(jnp.isnan(delta), -jnp.inf, delta)
            accept_prob = jnp.clip(delta, a_max=1.0)
            u = jax.random.uniform(opt_state.rng_key)
            accepted = accept_prob > u

            # x_new = x_new if accepted else x #accept/reject
            x_new = jnp.where(accepted, x_new, x)
        return x_new, opt_state, {"accepted": accepted, "accept_prob": accept_prob}

    return step, init_opt_state


from tqdm import tqdm
def gibbs_sampler(*, seed, log_probs, data, step_sizes, num_components,
                     mu_0, v_0, alpha,
                     a_0, b_0,
                     mh=False, n_iters=10000, burn_in=1000):

    rng = random.PRNGKey(seed)
    init_key, key = random.split(rng, 2)

    z_key, pi_key, mu_key, sigma_key = jax.random.split(init_key, 4)

    k, n = num_components, data.shape[0]

    pi0 = alpha / k
    z0 = tfd.Categorical(probs=pi0).sample(seed=z_key, sample_shape=(n,))
    mu0 = tfd.MultivariateNormalDiag(mu_0, jnp.sqrt(v_0)).sample(seed=mu_key)
    # sigma0 = tfd.InverseGamma(a_0, b_0).sample(seed=sigma_key)
    sigma0 = v_0

    z_log_prob_fn = log_probs["z"](data)
    mu_log_prob_fn = log_probs["mu"](data, mu_0, v_0, num_components)
    pi_log_prob_fn = log_probs["pi"](alpha, num_components)
    sigma_log_prob_fn = log_probs["sigma"](data, a_0, b_0, num_components)

    z_kernel_no_mh, z_opt_state_no_mh = get_discrete_kernel(seed, optax.constant_schedule(step_sizes["z"]),
                                                            z_log_prob_fn, z0[:,None],
                                                            dim=1, num_cls=num_components, cat=True, mh=False)

    z_kernel_mh, z_opt_state_mh = get_discrete_kernel(seed, optax.constant_schedule(step_sizes["z"]), z_log_prob_fn,
                                                         z0[:,None], dim=1, num_cls=num_components, cat=True, mh=True)

    pi_kernel_no_mh, pi_opt_state_no_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["pi"]),
                                                                pi_log_prob_fn, pi0, mh=False)

    pi_kernel_mh, pi_opt_state_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["pi"]),
                                                          pi_log_prob_fn, pi0, mh=True)

    mu_kernel_no_mh, mu_opt_state_no_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["mu"]),
                                                                mu_log_prob_fn, mu0, mh=False)

    mu_kernel_mh, mu_opt_state_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["mu"]),
                                                          mu_log_prob_fn, mu0, mh=True)

    sigma_kernel_no_mh, sigma_opt_state_no_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["sigma"]), sigma_log_prob_fn,
                                                                      sigma0, mh=False)

    sigma_kernel_mh, sigma_opt_state_mh = get_continuous_kernel(seed, optax.constant_schedule(step_sizes["sigma"]),
                                                                sigma_log_prob_fn, sigma0, mh=True)

    z_kernel_no_mh = jax.jit(z_kernel_no_mh)
    z_kernel_mh = jax.jit(z_kernel_mh)
    pi_kernel_no_mh = jax.jit(pi_kernel_no_mh)
    pi_kernel_mh = jax.jit(pi_kernel_mh)
    mu_kernel_no_mh = jax.jit(mu_kernel_no_mh)
    mu_kernel_mh = jax.jit(mu_kernel_mh)
    sigma_kernel_no_mh = jax.jit(sigma_kernel_no_mh)
    sigma_kernel_mh = jax.jit(sigma_kernel_mh)

    z_kernel = z_kernel_no_mh
    pi_kernel = pi_kernel_no_mh
    mu_kernel = mu_kernel_no_mh
    sigma_kernel = sigma_kernel_no_mh

    z_opt_state = z_opt_state_no_mh
    pi_opt_state = pi_opt_state_no_mh
    mu_opt_state = mu_opt_state_no_mh
    sigma_opt_state = sigma_opt_state_no_mh

    switch_to_mh = False
    state = {"pi": pi0, "z": z0, "mu": mu0,  "sigma": sigma0}
    samples = {k: [] for k in state.keys()}
    for k, v in state.items():
        samples[k].append(v)

    accept_probs = {"pi": np.zeros((n_iters)), "mu": np.zeros(n_iters), "sigma": np.zeros(n_iters),
                    "z": np.zeros(n_iters)}

    # sigma = jnp.array([1.0, 2.0, 1.0])
    # sigma = sigma0
    for i in tqdm(range(n_iters)):
        if i > burn_in and mh and not switch_to_mh:
            z_kernel = z_kernel_mh
            pi_kernel = pi_kernel_mh
            mu_kernel = mu_kernel_mh
            sigma_kernel = sigma_kernel_mh

            # z_opt_state = z_opt_state_mh
            # pi_opt_state = pi_opt_state_mh
            # mu_opt_state = mu_opt_state_mh
            # sigma_opt_state = sigma_opt_state_mh

            switch_to_mh = True

        z, z_opt_state, z_mh_info = z_kernel(state["z"][:,None], state, z_opt_state)
        z = z.squeeze()
        pi, pi_opt_state, pi_mh_info = pi_kernel(state["pi"], state, pi_opt_state)
        mu, mu_opt_state, mu_mh_info = mu_kernel(state["mu"], state, mu_opt_state)
        sigma, sigma_opt_state, sigma_mh_info = sigma_kernel(state["sigma"], state, sigma_opt_state)

        #update accept probs
        accept_probs["pi"][i] = pi_mh_info["accept_prob"]
        accept_probs["mu"][i] = mu_mh_info["accept_prob"]
        accept_probs["sigma"][i] = sigma_mh_info["accept_prob"]
        accept_probs["z"][i] = z_mh_info["accept_prob"]

        state = {"pi": pi, "z": z, "mu": mu,  "sigma": sigma}
        if i > burn_in:
            for k, v in state.items():
                samples[k].append(v)

    return samples, accept_probs