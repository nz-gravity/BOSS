from collections import namedtuple
from scipy.stats import median_abs_deviation
from backgrounds import StochasticBackgroundResponse, utils, signal
from numpyro.infer.util import log_density
import matplotlib.pyplot as plt
import corner
import jax
from scipy.interpolate import BSpline
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import summary
import healpy as hp
from backgrounds import StochasticBackgroundResponse

np.random.seed(1)
pi=np.pi
c=299792458
L=2.5e9
EPS = 1e-300
EXP_CLIP = 80.0  # clipping exponent prevents overflow in exp()

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

#Credible intervals
def compute_ci(psds):
    """
    Calculates the 5th, 50th, and 95th percentiles (pointwise)
    for a distribution of samples.
    psds: spectral density
    output: credible intervals
    """
    CI = namedtuple('CI', ['u05', 'u95', 'med'])
    lo, med, hi = np.nanpercentile(psds, [5, 50, 95], axis=0)

    return CI(u05=lo, u95=hi, med=med)

#signal to noise ratio:
def _const(a,b,n):
    return (a-b)/n

def _squaredratio(a,b):
    return ((a)**2)/((b)**2)

def computesnr(psd_signal, psd_noise, upper, lower, T_sec):  # note use PSD not log PSD
    """
    Calculates the SNR for a Stochastic GW Background  (A and E channels and assuming A=E) based on arXiv:1908.00546 (mid-point rule approximation for the integral).
    """
    return np.sqrt(2)*np.sqrt(T_sec* sum(_squaredratio(psd_signal,psd_noise)) * _const(a=upper,b=lower,n=len(psd_signal)))


#Relative integrated absolute error (RIAE)
def _absdiff(a,b):
    return abs(a-b)

def compute_rel_iae(psd, truepsd, upper, lower):  # note use PSD not log PSD
    return sum(_absdiff(psd,truepsd))/sum(truepsd)

# LISA noise:
def testmass_psd(atm, f):
    '''
    Testmass theoretical spectrum
    :param atm: testmass parameter
    :param f: frequencies
    :return: testmass PSD
    '''
    f = np.asarray(f, float)
    f = np.clip(f, 1e-12, None)  # dividing by zero
    return (atm)**2 * (1 + (0.4e-3/f)**2) * (1 + (f/8e-3)**4) * (1/(2*pi*f*c))**2

def oms_psd(aoms, f):
    '''
    OMS theoretical spectrum
    :param aoms: OMS parameter
    :param f: frequencies
    :return: OMS PSD
    '''
    f = np.asarray(f, float)
    f = np.clip(f, 1e-12, None) #dividing by zero
    return (aoms)**2 * (1 + (2e-3/f)**4) * (2*pi*f/c)**2

def x_phase(f, L=L, c=c):
    return 2.0 * np.pi * f * (L / c)

def common(f):
    x = x_phase(f)
    return 16.0 * (np.sin(x)**2) * (np.sin(2.0*x)**2)

def A_psd(atm, aoms, f):
    '''
    Theoretical noise spectrum of TDI A channel (Assuming same for E channel)
    :param atm: testmass parameter
    :param aoms: oms parameter
    :param f: frequencies
    :return: TDI A PSD
    '''
    x = x_phase(f)
    tm = 4.0 * common(f) * (3.0 + 2.0*np.cos(x) + np.cos(2.0*x)) * testmass_psd(atm, f)
    om = 2.0 * common(f) * (2.0 + np.cos(x)) * oms_psd(aoms, f)
    return tm + om

def T_psd(atm, aoms, f):
    '''
    Theoretical spectrum of TDI T channel (assuming no signal component in this channel)
    :param atm: testmass parameter
    :param aoms: oms parameter
    :param f: frequencies
    :return: TDI T PSD
    '''

    x = x_phase(f)
    tm = 32.0 * common(f) * (np.sin(0.5*x)**4) * testmass_psd(atm, f)
    om = 4.0 * common(f) * (1.0 - np.cos(x)) * oms_psd(aoms, f)
    return tm + om

def A_psd_tmoms(stm, soms, f):
    '''
    Theoretical TDI A channel noise spectral density in terms of testmass and oms spectral densities (to be used later in the spline analysis)
    :param stm: testmass psd
    :param soms: oms psd
    :param f: frequencies
    :return: TDI A PSD
    '''
    x = x_phase(f)
    tm = 4.0 * common(f) * (3.0 + 2.0*np.cos(x) + np.cos(2.0*x)) * stm
    om = 2.0 * common(f) * (2.0 + np.cos(x)) * soms
    return tm + om

def T_psd_tmoms(stm, soms, f):
    '''
    Theoretical TDI T channel spectral density in terms of testmass and oms spectral densities (to be used later in the spline analysis)
    :param stm: testmass psd
    :param soms: oms psd
    :param f: frequencies
    :return: TDI T PSD
    '''

    x = x_phase(f)
    tm = 32.0 * common(f) * (np.sin(0.5*x)**4) * stm
    om = 4.0 * common(f) * (1.0 - np.cos(x)) * soms
    return tm + om


#SGWB function (power law)
def sgwb_gen_fun(f_fit,n,f0,Omega0,tdi_response):
    '''
    Stochastic gravitational wave background (SGWB) PSD function (power law)
    :param f_fit: frequencies
    :param n: power
    :param f0: pivot frequency
    :param Omega0: amplitude
    :param tdi_response: LISA GW response matrix (known)
    :return: SGWB PSD
    '''
    s_h = signal.sgwb_psd(f_fit, spec_index=n, freq0=f0, omega_gw=Omega0)
    tdi_covariance = (tdi_response.T * s_h).T
    sig_A_spec=abs(tdi_covariance[:, 0, 0])
    return sig_A_spec

#SGWB function (gaussian bump)
def omega_gw_gaussian_bump(f, Omega_star=1e-11, f_star=3e-3, sigma=0.5):
    """
    Gaussian-bump SGWB energy density.

    Omega_gw(f) = Omega_star * exp[(log(f/f_star)/sigma)^2]
    """
    f = np.asarray(f, dtype=float)
    return Omega_star * np.exp(-(np.log10(f / f_star) / sigma)**2)


def strain_psd_from_omega(f, Omega_gw, H0_km_s_Mpc=67.8):
    """
    Strain PSD S_h(f).

    S_h(f) = 3 H0^2 / (4 pi^2 f^3) * Omega_gw(f)
    """
    f = np.asarray(f, dtype=float)

    Mpc_in_km = 3.0856775814913673e19
    H0 = H0_km_s_Mpc / Mpc_in_km  # s^{-1}

    return (3 * H0**2 / (4 * np.pi**2 * f**3)) * Omega_gw

def gb_gen_fun(f, Omega_star, f_star, sigma, tdi_response):
    '''
    Stochastic gravitational wave background (SGWB) PSD function (Gaussian bump)
    :param f: frequencies
    :param Omega_star: Amplitude
    :param f_star: central frequency
    :param sigma: width
    :param tdi_response: LISA GW response matrix (known)
    :return: SGWB PSD
    '''
    Omega_gw = omega_gw_gaussian_bump(
    f,
    Omega_star=Omega_star,
    f_star=f_star,
    sigma=sigma
    )
    Sh = strain_psd_from_omega(f, Omega_gw)
    tdi_covariance = (tdi_response.T * Sh).T
    sig_A_spec=abs(tdi_covariance[:, 0, 0])
    return sig_A_spec

##response
def get_response_matrix(freqs, t0, orbits_path):
    """
    Computes the AET response matrix at time t0.
    This code is taken from https://qbaghi.pages.in2p3.fr/backgrounds/notebooks/sgwb-response.html by Quentin Baghi
    """
    nside = 8
    npix = hp.nside2npix(nside)
    m = np.ones(npix) / np.sqrt(npix) * np.sqrt((4 * np.pi) / 2)# sky averaging
    sgwb_cls = StochasticBackgroundResponse(m, orbits=orbits_path)
    tdi_response = sgwb_cls.compute_tdi_kernel(fr=freqs, t0=t0, tdi_var='aet', gen='2.0', parallel=False)
    return tdi_response


#converting numpy arrays to jax arrays
def _convert_numpy_to_jax(x):
    return jnp.asarray(x)

#Bspline functions:
def create_bspline_basis(knots, degree, x_eval):
    """Create B-spline basis matrix"""
    knots_extended = np.concatenate([
        np.repeat(knots[0], degree),
        knots,
        np.repeat(knots[-1], degree)
    ])

    n_basis = len(knots) + degree - 1
    basis_matrix = np.zeros((len(x_eval), n_basis))

    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        bspline = BSpline(knots_extended, coeffs, degree)
        basis_matrix[:, i] = bspline(x_eval)

    return basis_matrix


def create_penalty_matrix(n_basis, order=1):
    """Create penalty matrix for smoothness"""
    if order == 0:
        return np.eye(n_basis)

    D = np.eye(n_basis)
    for _ in range(order):
        D_new = np.zeros((D.shape[0] - 1, D.shape[1]))
        for i in range(D.shape[0] - 1):
            D_new[i] = D[i + 1] - D[i]
        D = D_new

    return D.T @ D


def create_log_spaced_knots(n_knots, freq_min, freq_max):
    """Create logarithmically spaced knots"""
    if n_knots < 2:
        raise ValueError("Need at least 2 knots")

    if freq_min <= 0:
        raise ValueError("freq_min must be > 0 for log-spaced knots")

    knot_freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_knots)
    knots_normalized = (knot_freqs - freq_min) / (freq_max - freq_min)

    return np.clip(knots_normalized, 0.0, 1.0), knot_freqs


def estimate_lambda_from_parametric(
    estimated,
    basis_matrix,
    penalty_matrix,
    tau=1e-4,
    eps=1e-8
):
    '''
    Estimating spline parameters from a given spectrum
    '''
    y = np.log(estimated)
    B = basis_matrix
    P = penalty_matrix

    BtB = B.T @ B
    mat = BtB + tau * P + eps * np.eye(B.shape[1])
    rhs  = B.T @ y

    lambda_init = np.linalg.solve(mat, rhs)
    return _convert_numpy_to_jax(lambda_init)

def knots_and_spline_mat(n_knots,f_fit,degree):
    '''Creating knots vector and spline matrix '''
    knots_normalized, knot_freqs = create_log_spaced_knots(
        n_knots, f_fit[0], f_fit[-1]
    )

    basis_matrix = create_bspline_basis(
        knots_normalized,
        degree,
        (f_fit - f_fit[0]) /
        (f_fit[-1] - f_fit[0])
    )
    return jnp.asarray(knots_normalized), jnp.asarray(basis_matrix)

def _reg_sym_pan(penalty_matrix,n_weights):
    # regularise + symmetrise penalties (helps MultivariateNormal(precision_matrix=...))
    P_reg = penalty_matrix + 1e-6 * jnp.eye(n_weights)
    P_reg = _convert_numpy_to_jax(0.5 * (P_reg + P_reg.T))
    return P_reg

@jax.jit
def _safe_exp(x):
    return jnp.exp(jnp.clip(x, -EXP_CLIP, EXP_CLIP))

#Spline PSD:
def spline_psd(basis_matrix,weights):
    return _safe_exp(basis_matrix @ weights)

#sgwb function (non-parametric)
def sgwb_nonparam(basis_matrix_sgwb,weights_sgwb,res_mat):
    '''
    SGWB non-parametric PSD function (splines)
    :param basis_matrix_sgwb: basis matrix for SGWB
    :param weights_sgwb: spline parameters sgwb
    :param res_mat: LISA GW response matrix (known)
    :return: SGWB PSD
    '''

    sh_psd =spline_psd(basis_matrix_sgwb,weights_sgwb)
    tdi_covariance = (res_mat.T * sh_psd).T
    sig_A_full= abs(tdi_covariance[:, 0, 0])
    return sig_A_full

#initial variables class
class init_var:
    def __init__(self,
              n_knots,
              f,
              degree,
              penalty_order,
              tm_spec,
              oms_spec,
              factor,
              log_pdgrm,
              log_pdgrm_E,
              log_pdgrm_T,
              phi_val,
              num_chains,
              phi_val_oms=10000,
              phi_val_sgwb=10,
              f0=3.16e-3,
              res_mat=None,
              np_sig=False,
              init_alpha=-3,
              init_log_10_omega=-13,
                 ):
        self.num_chains = num_chains
        self.n_knots = n_knots
        self.n_weights = n_knots + degree - 1
        self.f = f
        self.degree = degree
        self.penalty_order = penalty_order
        self.tm_spec = tm_spec
        self.oms_spec = oms_spec
        self.factor = factor
        self.log_pdgrm = log_pdgrm
        self.log_pdgrm_E = log_pdgrm_E
        self.log_pdgrm_T = log_pdgrm_T
        self.f0 = f0
        self.res_mat = res_mat
        self.phi_val = phi_val
        self.phi_val_oms = phi_val_oms
        self.phi_val_sgwb = phi_val_sgwb

        log_fac=np.log(self.factor)

        self.knots, self.basis_matrix = knots_and_spline_mat(n_knots=self.n_knots, f_fit=self.f, degree=self.degree)
        self.knots_oms, self.basis_matrix_oms = knots_and_spline_mat(n_knots=self.n_knots, f_fit=self.f, degree=self.degree)
        self.n_basis = self.basis_matrix.shape[1]
        penalty_matrix = create_penalty_matrix(self.n_basis, self.penalty_order)
        penalty_matrix_oms = create_penalty_matrix(self.n_basis, self.penalty_order)
        if tm_spec is None:
            # noise param (theoretical noise model)
            self.atm = 3e-15
            self.tm_spec = testmass_psd(self.atm, self.f)

        if oms_spec is None:
            # noise param (theoretical noise model)
            self.aoms = 15e-12
            self.oms_spec = oms_psd(self.aoms, self.f)

        # initial estimate:
        self.init_weights = estimate_lambda_from_parametric(
            estimated=self.tm_spec * self.factor,
            basis_matrix=self.basis_matrix,
            penalty_matrix=penalty_matrix,
            tau=1e-4
        )

        self.init_weights_oms = estimate_lambda_from_parametric(
            estimated=self.oms_spec * self.factor,
            basis_matrix=self.basis_matrix_oms,
            penalty_matrix=penalty_matrix_oms,
            tau=1e-4
        )
        self.n_weights = self.basis_matrix.shape[1]
        self.P_reg = _reg_sym_pan(penalty_matrix, self.n_weights)
        self.P_reg_oms = _reg_sym_pan(penalty_matrix_oms, self.n_weights)

        self.log_pdgrm = _convert_numpy_to_jax(self.log_pdgrm)
        self.log_pdgrm_E = _convert_numpy_to_jax(self.log_pdgrm_E)
        self.log_pdgrm_T = _convert_numpy_to_jax(self.log_pdgrm_T)
        self.basis_matrix = _convert_numpy_to_jax(self.basis_matrix)
        self.basis_matrix_oms = _convert_numpy_to_jax(self.basis_matrix_oms)
        self.f = _convert_numpy_to_jax(self.f)
        if np_sig:
            self.knots_sgwb, self.basis_matrix_sgwb = knots_and_spline_mat(n_knots=self.n_knots, f_fit=self.f, degree=self.degree)
            penalty_matrix_sgwb = create_penalty_matrix(self.n_basis, self.penalty_order)
            self.P_reg_sgwb = _reg_sym_pan(penalty_matrix_sgwb, self.n_weights)
            self.basis_matrix_sgwb = _convert_numpy_to_jax(self.basis_matrix_sgwb)
            #SGWB prior located at kinetic energy-dominated phase SGWB.
            s_h = signal.sgwb_psd(self.f, spec_index=1, freq0=f0, omega_gw=1e-13)
            self.weights_sgwb_loc = estimate_lambda_from_parametric(
                estimated=s_h * self.factor,
                basis_matrix=self.basis_matrix_sgwb,
                penalty_matrix=penalty_matrix_sgwb,
                tau=1e-4
            )
            self.lambda_init_sgwb= estimate_lambda_from_parametric(
                estimated=signal.sgwb_psd(self.f, spec_index=init_alpha, freq0=f0, omega_gw=10**init_log_10_omega) * self.factor,
                basis_matrix=self.basis_matrix_sgwb,
                penalty_matrix=penalty_matrix_sgwb,
                tau=1e-4
            )



def _lighten(hex_color, amount=0.5):
    """Lightens a hex color by blending with white."""
    import matplotlib.colors as mc
    import colorsys
    c = colorsys.rgb_to_hls(*mc.to_rgb(hex_color))
    lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return lightened


#output class
class output:
    def __init__(self,init_object, sampler_object, signal_param=False, signal_nparam=False):
        self.n_knots = init_object.n_knots
        self.n_weights = init_object.n_weights
        self.f = init_object.f
        self.degree = init_object.degree
        self.factor = init_object.factor
        self.penalty_order = init_object.penalty_order
        self.log_pdgrm = init_object.log_pdgrm
        self.log_pdgrm_E = init_object.log_pdgrm_E
        self.log_pdgrm_T = init_object.log_pdgrm_T
        self.f0 = init_object.f0
        self.res_mat = init_object.res_mat
        self.knots, self.basis_matrix=init_object.knots, init_object.basis_matrix
        self.knots_oms, self.basis_matrix_oms=init_object.knots_oms, init_object.basis_matrix_oms
        self.penalty_matrix = init_object.P_reg
        self.penalty_matrix_oms = init_object.P_reg_oms
        self.init_weights=init_object.init_weights
        self.init_weights_oms=init_object.init_weights_oms
        self.phi_val=init_object.phi_val
        self.phi_val_oms = init_object.phi_val_oms
        self.phi_val_sgwb = init_object.phi_val_sgwb
        if init_object.num_chains>1:
            orbital_samples = sampler_object.get_samples(group_by_chain=True)
            stats = summary(orbital_samples)

        orbital_samples = sampler_object.get_samples()
        self.weights_samples = np.asarray(orbital_samples["weights"])  # (S, K)
        self.weights_samples_oms = np.asarray(orbital_samples["weights_oms"])  # (S, K)
        tm_psd_samples_scaled = np.exp(self.weights_samples @ np.asarray(init_object.basis_matrix).T)  # (S, nfreq)
        oms_psd_samples_scaled = np.exp(self.weights_samples_oms @ np.asarray(init_object.basis_matrix_oms).T)  # (S, nfreq)

        self.tm_psd_samples = tm_psd_samples_scaled * (1/self.factor)
        self.oms_psd_samples = oms_psd_samples_scaled * (1/self.factor)

        self.A_spec_samples = A_psd_tmoms(stm=self.tm_psd_samples, soms=self.oms_psd_samples, f=self.f)
        self.T_spec_samples = T_psd_tmoms(stm=self.tm_psd_samples, soms=self.oms_psd_samples, f=self.f)

        self.ci_A = compute_ci(self.A_spec_samples)
        self.ci_T = compute_ci(self.T_spec_samples)
        self.ci_TM = compute_ci(self.tm_psd_samples)
        self.ci_OM = compute_ci(self.oms_psd_samples)
        if signal_param:
            if init_object.num_chains>1:
                self.r_hat = dict(alpha_sgwb=stats['alpha_sgwb']['r_hat'],
                                  log10_omega=stats['log10_omega']['r_hat'],
                                  weights=stats['weights']['r_hat'],
                                  weights_oms=stats['weights_oms']['r_hat'])
                self.ess = dict(alpha_sgwb=stats['alpha_sgwb']['n_eff'],
                                log10_omega=stats['log10_omega']['n_eff'],
                                weights=stats['weights']['n_eff'],
                                weights_oms=stats['weights_oms']['n_eff'])

            self.log10_omega_samps = np.asarray(orbital_samples["log10_omega"])
            self.alpha_samps = np.asarray(orbital_samples["alpha_sgwb"])

            ns = self.log10_omega_samps.shape[0]
            nf = self.f.shape[0]

            # --- SGWB PSD curve for each posterior draw ---
            self.sgwb_samples = np.empty((ns, nf), dtype=float)
            for i, (lw, a) in enumerate(zip(self.log10_omega_samps, self.alpha_samps)):
                omega = 10.0 ** lw
                sigA_i=sgwb_gen_fun(f_fit=self.f, n=a, f0=self.f0, Omega0=omega, tdi_response=self.res_mat)
                self.sgwb_samples[i, :] = np.asarray(sigA_i)
            self.ci_sgwb=compute_ci(self.sgwb_samples)

        if signal_nparam:
            if init_object.num_chains>1:
                self.r_hat = dict(weights_sgwb=stats['weights_sgwb']['r_hat'],
                                  weights=stats['weights']['r_hat'],
                                  weights_oms=stats['weights_oms']['r_hat'])
                self.ess = dict(weights_sgwb=stats['weights_sgwb']['n_eff'],
                                weights=stats['weights']['n_eff'],
                                weights_oms=stats['weights_oms']['n_eff'])

            self.knots_sgwb, self.basis_matrix_sgwb = init_object.knots_sgwb, init_object.basis_matrix_sgwb
            self.penalty_matrix_sgwb = init_object.P_reg_sgwb
            self.weights_sgwb_loc=init_object.weights_sgwb_loc
            self.weights_samples_sgwb = np.asarray(orbital_samples["weights_sgwb"])  # (S, K)

            sig_psd_est = np.exp(self.weights_samples_sgwb @ np.asarray(self.basis_matrix_sgwb).T)
            sigA_samples = np.empty((sig_psd_est.shape[0], sig_psd_est.shape[1]), dtype=float)

            for i in range(0, sig_psd_est.shape[0]):
                tdi_covariance = (self.res_mat.T * sig_psd_est[i]).T
                sigA_i = abs(tdi_covariance[:, 0, 0])
                sigA_samples[i, :] = np.asarray(sigA_i * (1 / init_object.factor))
            self.sgwb_psd_samples = sigA_samples
            self.ci_sgwb = compute_ci(self.sgwb_psd_samples)

    def plot_signal_A_noise(
                self,
                A_spec=None,
                sig_A_true=None,
                outpath=None,
                label="SGWB estimated",
                ylim=None,
                dpi=300,
        ):

            plt.style.use('default')
            plt.figure(figsize=(7, 4.5))
            plt.plot(self.f,np.exp(self.log_pdgrm)*(1/self.factor), label="A periodogram", alpha=0.3)
            plt.fill_between(self.f, self.ci_A.u05, self.ci_A.u95,
                             color='red', alpha=0.8, label='estimated A noise')
            if A_spec is not None:
                plt.plot(self.f, A_spec, label="A", color='k', linestyle="--")
            plt.fill_between(self.f, self.ci_sgwb.u05, self.ci_sgwb.u95, alpha=0.8, color='orange', label='estimated signal')
            if sig_A_true is not None:
                plt.plot(self.f, sig_A_true, linestyle="--", label="SGWB true", color='blue')

            plt.xlabel("f [Hz]")
            plt.ylabel("PSD [1/Hz]")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()

            if ylim is None:
                ylim=[min(self.ci_A.u05), max(self.ci_A.u95)]
            if outpath is not None:
                plt.savefig(f"{outpath}/sig_sep.pdf", dpi=dpi)
            plt.ylim(ylim)
            plt.show()

    def plot_corner(self,
                    true_vals=None,
                    filename=None,
                    dire=None,
                    dpi=150,
                    max_samples=1500,
                    seed=42):

        has_param = hasattr(self, 'alpha_samps') and hasattr(self, 'log10_omega_samps')
        has_nparam = hasattr(self, 'weights_samples_sgwb')
        n_knots = self.weights_samples.shape[1]

        COLOR_SIGNAL = "#E67E22"  # orange  — SGWB / GW parameters
        COLOR_TM = "#7B2D8B"  # purple  — test-mass spline weights
        COLOR_OMS = "#009B8D"  # teal    — OMS spline weights

        # Downsample
        n_total = self.weights_samples.shape[0]
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=min(max_samples, n_total), replace=False)
        idx.sort()
        print(f"Using {len(idx)} / {n_total} samples.")
        def _corner(samps, labels, truths=None, fname=None, color=COLOR_SIGNAL):
            fig = corner.corner(
                samps,
                labels=labels,
                truths=truths,
                truth_color="#111111",
                quantiles=[0.16, 0.5, 0.84],
                label_kwargs={"fontsize": 9},
                color=color,
                hist_kwargs={"lw": 1.2},
                plot_contours=True,
                fill_contours=True,
                levels=[0.68, 0.95],
                contourf_kwargs={"alpha": 0.25,
                                 "colors": [_lighten(color, 0.6),
                                            _lighten(color, 0.3)]},
                smooth=1.0,
                smooth1d=1.0,
                bins=35,
                max_n_ticks=3,
            )
            if fname is not None and dire is not None:
                fig.savefig(f"{dire}/{fname}.pdf", bbox_inches='tight', dpi=dpi)
                print(f"Saved {fname}.pdf")
            plt.show()
            plt.close(fig)

        base = filename or "corner"
        if has_param:
            _corner(
                samps=np.column_stack([self.alpha_samps[idx],
                                       self.log10_omega_samps[idx]]),
                labels=[r'$\alpha$', r'$\log_{10}\Omega$'],
                truths=[true_vals.get('alpha'),
                        true_vals.get('log10_omega')] if true_vals else None,
                fname=f"{base}_gw_params",
                color=COLOR_SIGNAL,
            )
            _corner(
                samps=self.weights_samples[idx],
                labels=[fr'$\lambda^{{TM}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_tm_weights",
                color=COLOR_TM,
            )
            _corner(
                samps=self.weights_samples_oms[idx],
                labels=[fr'$\lambda^{{OMS}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_oms_weights",
                color=COLOR_OMS,
            )

        elif has_nparam:
            _corner(
                samps=self.weights_samples_sgwb[idx],
                labels=[fr'$\lambda^{{S}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_sgwb_weights",
                color=COLOR_SIGNAL,
            )
            _corner(
                samps=self.weights_samples[idx],
                labels=[fr'$\lambda^{{TM}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_tm_weights",
                color=COLOR_TM,
            )
            _corner(
                samps=self.weights_samples_oms[idx],
                labels=[fr'$\lambda^{{OMS}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_oms_weights",
                color=COLOR_OMS,
            )

        else:
            _corner(
                samps=self.weights_samples[idx],
                labels=[fr'$\lambda^{{TM}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_tm_weights",
                color=COLOR_TM,
            )
            _corner(
                samps=self.weights_samples_oms[idx],
                labels=[fr'$\lambda^{{OMS}}_{{{i + 1}}}$' for i in range(n_knots)],
                fname=f"{base}_oms_weights",
                color=COLOR_OMS,
            )

    def plot_sgwb_param_trace(self,n=None, Omega0=None, dpi=300, outpath=None):
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 1, 1)
        plt.plot(10 ** self.log10_omega_samps, ".", alpha=0.3)
        if Omega0 is not None:
            plt.axhline(Omega0, color="k")
        plt.ylabel("Omega")
        plt.xlabel("iteration")
        plt.subplot(2, 1, 2)
        plt.plot(self.alpha_samps, ".", alpha=0.3)
        if n is not None:
            plt.axhline(n, color="k")
        plt.ylabel("n")
        plt.xlabel("iteration")
        if outpath is not None:
            plt.savefig(f"{outpath}/sgwb_param_trace.pdf")
        plt.show()

        phys_samples = np.column_stack([self.alpha_samps, 10 ** self.log10_omega_samps])
        corner.corner(phys_samples, labels=[r"", r""], truths=[n, Omega0])
        if outpath is not None:
            plt.savefig(f"{outpath}/sgwb_param_corner.pdf", dpi=dpi)
        plt.show()


    def plot_noises(self,A_spec=None,T_spec=None,tm_spec=None,oms_spec=None,outpath=None,dpi=300,ylim=None,):
        plt.figure(figsize=(8, 4))
        plt.fill_between(self.f, self.ci_A.u05, self.ci_A.u95,
                         color='orange', alpha=0.3, linewidth=0)
        plt.loglog(self.f, self.ci_A.med, label="A_estimated", color="orange")
        if A_spec is not None:
            plt.loglog(self.f, A_spec, label="A_true", linestyle="--", color='blue')
        plt.fill_between(self.f, self.ci_T.u05,self.ci_T.u95,
                         color='red', alpha=0.3, linewidth=0)
        plt.loglog(self.f, self.ci_T.med, label="T_estimated", color="red")
        if T_spec is not None:
            plt.loglog(self.f, T_spec, label="T_true", linestyle="--", color='black')
        plt.fill_between(self.f, self.ci_TM.u05, self.ci_TM.u95,
                         color='pink', alpha=0.3, linewidth=0)
        plt.loglog(self.f, self.ci_TM.med, label="TM_estimated", color="pink")
        if tm_spec is not None:
            plt.loglog(self.f, tm_spec, label='testmass psd true', linestyle='--', color='purple')
        plt.fill_between(self.f, self.ci_OM.u05, self.ci_OM.u95,
                         color='brown', alpha=0.3, linewidth=0)
        plt.loglog(self.f, self.ci_OM.med, label="OMS_estimated", color="brown")
        if oms_spec is not None:
            plt.loglog(self.f, oms_spec, label='oms psd true', linestyle='--', color='green')
        if ylim is None:
            ylim = [min(self.ci_T.med), max(self.ci_A.med)]

        plt.ylim(ylim)
        plt.xlabel("f [Hz]")
        plt.ylabel("PSD [1/Hz]")
        plt.legend()
        if outpath is not None:
            plt.savefig(f"{outpath}/estimatednoises.pdf", dpi=dpi)
        plt.show()



#theoretical models (jax compatable version)
@jax.jit
def _safe_log(x):
    return jnp.log(jnp.maximum(x, EPS))


@jax.jit
def x_phase_jax(f):
    return 2.0 * jnp.pi * f * (L / c)

@jax.jit
def common_jax(f):
    x = x_phase_jax(f)
    return 16.0 * jnp.sin(x) ** 2 * jnp.sin(2.0 * x) ** 2

@jax.jit
def A_psd_tmoms_jax(stm, soms, f):
    x = x_phase_jax(f)
    tm = 4.0 * common_jax(f) * (3.0 + 2.0 * jnp.cos(x) + jnp.cos(2.0 * x)) * stm
    om = 2.0 * common_jax(f) * (2.0 + jnp.cos(x)) * soms
    return tm + om

@jax.jit
def T_psd_tmoms_jax(stm, soms, f):
    x = x_phase_jax(f)
    tm = 32.0 * common_jax(f) * (jnp.sin(0.5 * x) ** 4) * stm
    om = 4.0 * common_jax(f) * (1.0 - jnp.cos(x)) * soms
    return tm + om


@jax.jit
def loglike_whittle(logS, logI):
    '''
    Whittle likelihood function -sum(logS + I/S)
    :param logS: log spectral density
    :param logI: log periodogram
    :return: log likelihood
    '''
    z = jnp.clip(logI - logS, -EXP_CLIP, EXP_CLIP)
    return -jnp.sum(logS + jnp.exp(z))

def loglikelihood(weights, weights_oms,
                  basis_matrix, basis_matrix_oms,
                  log_pdgrm,log_pdgrm_E, log_pdgrm_T, f,
                  omega_ref, alpha_sgwb,
                  num_segments, res_mat,f0,factor):
    '''
    Gamma likelihood function (parametric (power law) signal and non-parametric noise models)
    :param weights: testmass spline weights
    :param weights_oms: oms spline weights
    :param basis_matrix: testmass basis matrix
    :param basis_matrix_oms: oms basis matrix
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param omega_ref: log amplitude SGWB
    :param alpha_sgwb: spectral index SGWB
    :param num_segments: number of segments
    :param res_mat: LISA GW response matrix
    :param f0: pivot frequency
    :param factor: factor
    :return: log likelihood for three channels
    '''
    tm_psd  = spline_psd(basis_matrix,weights)
    oms_psd = spline_psd(basis_matrix_oms,weights_oms)

    A_noise = A_psd_tmoms_jax(stm=tm_psd,  soms=oms_psd, f=f)          # scaled
    T_noise = T_psd_tmoms_jax(stm=tm_psd,  soms=oms_psd, f=f)

    sig_A  =  _convert_numpy_to_jax(sgwb_gen_fun(f_fit=f,n=alpha_sgwb,f0=f0,Omega0=omega_ref,tdi_response=res_mat))


    log_A = _safe_log(A_noise + sig_A*factor)
    log_E = _safe_log(A_noise + sig_A*factor)
    log_T = _safe_log(T_noise)
    return num_segments * (
    loglike_whittle(log_A, log_pdgrm)
  + loglike_whittle(log_E, log_pdgrm_E)
  + loglike_whittle(log_T, log_pdgrm_T)
)


def bayesian_model(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,res_mat,f0,
    num_segments,
    factor,
    phi_val,
    phi_val_oms,
    P_reg,
    P_reg_oms,
    init_weights,
    init_weights_oms,
    basis_matrix,
    basis_matrix_oms,
):
    '''Bayesian model: parametric power law SGWB model and spline LISA noise model.'''
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )
    # Uniform priors
    log10_omega = numpyro.sample("log10_omega", dist.Uniform(-20.0, -9.0))
    omega_ref   = 10.0 ** log10_omega

    alpha_sgwb  = numpyro.sample("alpha_sgwb", dist.Uniform(-5.0, 5.0))

    ll = loglikelihood(
        weights=weights,
        weights_oms=weights_oms,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        omega_ref=omega_ref,
        alpha_sgwb=alpha_sgwb,
        num_segments=num_segments,
        res_mat=res_mat,
        f0=f0,
        factor=factor,
    )
    numpyro.factor("log_likelihood", ll)


# running the Bayesian model:
def mcmc(log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,res_mat,f0,
    n_knots= 10, degree= 3, penalty_order= 1,
    num_segments=1,
    factor=1,
    phi_val=10000,
    phi_val_oms=10000,
    num_warmup=500,num_samples=2500, num_chains=1, progress_bar=True,
    init_alpha=-3,
    init_log_10_omega=-13,
    tm_spec=None,
    oms_spec=None,
):
    '''
    MCMC function for SGWB separation from LISA noise. Signal model: parametric (power law), Noise model: Penalized P-splines
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param res_mat: LISA GW response matrix
    :param f0: pivot frequency
    :param n_knots: number of spline knots (assuming equal number of knots for all splines)
    :param degree: degree of the spline (default third order spline)
    :param penalty_order: penalty order of the penalty matrix (default 1)
    :param num_segments: number of segments
    :param factor: factor if normalized data
    :param phi_val: spline parmeter (phi)
    :param num_warmup: number of warmup (NUTS)
    :param num_samples: number of samples (NUTS)
    :param num_chains: number of chains (NUTS)
    :param progress_bar: if show progress
    :param init_alpha: starting value for spectral index
    :param init_log_10_omega: starting value for log amplitude
    :param tm_spec: testmass theoretical spectrum (for initial spline weights)
    :param oms_spec: oms theoretical spectrum (for initial spline weights)
    :return: mcmc output object containing estimated results and plots.
    '''
    myobj_spl=init_var(n_knots=n_knots,
              f=f,
              degree=degree,
              penalty_order=penalty_order,
              tm_spec=tm_spec,
              oms_spec=oms_spec,
              factor=factor,
              log_pdgrm=log_pdgrm,
              log_pdgrm_E=log_pdgrm_E,
              log_pdgrm_T=log_pdgrm_T,
              f0=f0,
              res_mat=res_mat,
              phi_val=phi_val,
              phi_val_oms=phi_val_oms,
              num_chains=num_chains,
              init_alpha=init_alpha,
              init_log_10_omega=init_log_10_omega,
              )

    nuts_kernel = NUTS(
        bayesian_model,
        init_strategy=init_to_value(values={
            "alpha_sgwb": init_alpha,
            "log10_omega": init_log_10_omega,
            "weights": myobj_spl.init_weights,
            "weights_oms": myobj_spl.init_weights_oms,
        }),
    )

    mcmc_bayes = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=progress_bar)

    print("Running MCMC...")
    mcmc_bayes.run(
        jax.random.PRNGKey(123),
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        res_mat=myobj_spl.res_mat,
        f0=myobj_spl.f0,
        num_segments=num_segments,
        factor=myobj_spl.factor,
        phi_val=myobj_spl.phi_val,
        phi_val_oms=myobj_spl.phi_val_oms,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
    )

    out_obj=output(init_object=myobj_spl, sampler_object=mcmc_bayes, signal_param=True, signal_nparam=False)
    return out_obj




#noise only model:
EPS = 1e-300
EXP_CLIP = 80.0  # clipping exponent prevents overflow in exp()

@jax.jit
def loglikelihood_noise(weights, weights_oms,
                  basis_matrix, basis_matrix_oms,
                  log_pdgrm,log_pdgrm_E, log_pdgrm_T, f,
                  num_segments):
    '''
    Gamma likelihood function (noise only model)
    :param weights: testmass spline weights
    :param weights_oms: oms spline weights
    :param basis_matrix: testmass basis matrix
    :param basis_matrix_oms: oms basis matrix
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param num_segments: number of segments
    :return: log likelihood for three channels
    '''
    tm_psd = spline_psd(basis_matrix, weights)
    oms_psd = spline_psd(basis_matrix_oms, weights_oms)

    A_noise = A_psd_tmoms_jax(stm=tm_psd, soms=oms_psd, f=f)  # scaled
    T_noise = T_psd_tmoms_jax(stm=tm_psd, soms=oms_psd, f=f)

    log_A = _safe_log(A_noise)
    log_E = _safe_log(A_noise)
    log_T = _safe_log(T_noise)

    return num_segments * (
    loglike_whittle(log_A, log_pdgrm)
  + loglike_whittle(log_E, log_pdgrm_E)
  + loglike_whittle(log_T, log_pdgrm_T)
)



def bayesian_model_noise(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,
    num_segments,
    phi_val,
    phi_val_oms,
    P_reg,
    P_reg_oms,
    init_weights,
    init_weights_oms,
    basis_matrix,
    basis_matrix_oms,
):
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )
    ll = loglikelihood_noise(
        weights=weights,
        weights_oms=weights_oms,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        num_segments=num_segments,
    )
    numpyro.factor("log_likelihood", ll)


# running the Bayesian model:
def mcmc_noise(log_pdgrm, log_pdgrm_E, log_pdgrm_T,
         f, n_knots=10, degree=3, penalty_order=1,
         num_segments=1,
         factor=1,
         phi_val=10000,
         phi_val_oms=10000,
         num_warmup=500, num_samples=2500, num_chains=1, progress_bar=True,
         tm_spec=None,
         oms_spec=None,
         ):
    '''
    MCMC function for noise only model (Assumes the data has just and no signal, will be used later for Bayes factor).
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param n_knots: number of spline knots (assuming equal number of knots for all splines)
    :param degree: degree of the spline (default third order spline)
    :param penalty_order: penalty order of the penalty matrix (default 1)
    :param num_segments: number of segments
    :param factor: factor if normalized data
    :param phi_val: testmass spline parmeter (phi)
    :param phi_val_oms: oms spline parmeter (phi)
    Note: for non-parametric model, different phi for two splines (testmass is restricted more for better signal detection).
    :param num_warmup: number of warmup (NUTS)
    :param num_samples: number of samples (NUTS)
    :param num_chains: number of chains (NUTS)
    :param progress_bar: if show progress
    :param tm_spec: testmass theoretical spectrum (for initial spline weights)
    :param oms_spec: oms theoretical spectrum (for initial spline weights)
    :return: mcmc output object containing estimated results and plots.
    '''

    myobj_spl=init_var(n_knots=n_knots,
              f=f,
              degree=degree,
              penalty_order=penalty_order,
              tm_spec=tm_spec,
              oms_spec=oms_spec,
              factor=factor,
              log_pdgrm=log_pdgrm,
              log_pdgrm_E=log_pdgrm_E,
              log_pdgrm_T=log_pdgrm_T,
              phi_val=phi_val,
              phi_val_oms=phi_val_oms,
              num_chains=num_chains)
    nuts_kernel = NUTS(
        bayesian_model_noise,
        init_strategy=init_to_value(values={
            "weights": myobj_spl.init_weights,
            "weights_oms": myobj_spl.init_weights_oms,
        }),
    )

    mcmc_bayes = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                      progress_bar=progress_bar)

    print("Running MCMC...")

    mcmc_bayes.run(
        jax.random.PRNGKey(123),
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        num_segments=num_segments,
        phi_val=phi_val,
        phi_val_oms=phi_val_oms,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
    )
    out_obj=output(init_object=myobj_spl, sampler_object=mcmc_bayes, signal_param=False, signal_nparam=False)
    return out_obj



#non-parametric signal model:
@jax.jit
def loglikelihood_nonparam(weights, weights_oms, weights_sgwb,
                               basis_matrix, basis_matrix_oms, basis_matrix_sgwb,
                               log_pdgrm, log_pdgrm_E, log_pdgrm_T, f,
                               num_segments, res_mat):
    '''
    Gamma likelihood function (non-parametric signal and non-parametric noise models)
    :param weights: testmass spline weights
    :param weights_oms: oms spline weights
    :param weights_sgwb: SGWB spline weights
    :param basis_matrix: testmass basis matrix
    :param basis_matrix_oms: oms basis matrix
    :param basis_matrix_sgwb: SGWB basis matrix
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param num_segments: number of segments
    :param res_mat: LISA GW response matrix
    :return: log likelihood for three channels
    '''

    # Noise
    tm_psd  = spline_psd(basis_matrix,weights)
    oms_psd = spline_psd(basis_matrix_oms,weights_oms)

    A_noise = A_psd_tmoms_jax(stm=tm_psd,  soms=oms_psd, f=f)          # scaled
    T_noise = T_psd_tmoms_jax(stm=tm_psd,  soms=oms_psd, f=f)

    #GW
    sig_A=sgwb_nonparam(basis_matrix_sgwb, weights_sgwb, res_mat)


    log_A = _safe_log(A_noise + sig_A)
    log_E = _safe_log(A_noise + sig_A)
    log_T = _safe_log(T_noise)

    return num_segments * (
        loglike_whittle(log_A, log_pdgrm)
      + loglike_whittle(log_E, log_pdgrm_E)
      + loglike_whittle(log_T, log_pdgrm_T)
    )



def bayesian_model_non_param(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,res_mat,
    num_segments,
    factor,
    phi_val,
    phi_val_oms,
    phi_val_sgwb,
    P_reg,
    P_reg_oms,
    P_reg_sgwb,
    init_weights,
    init_weights_oms,
    weights_sgwb_loc,
    basis_matrix,
    basis_matrix_oms,
    basis_matrix_sgwb,
):
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )

    weights_sgwb = numpyro.sample(
        "weights_sgwb",
        dist.MultivariateNormal(loc=weights_sgwb_loc, precision_matrix=phi_val_sgwb * P_reg_sgwb),
    )
    ll = loglikelihood_nonparam(
        weights=weights,
        weights_oms=weights_oms,
        weights_sgwb=weights_sgwb,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        basis_matrix_sgwb=basis_matrix_sgwb,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        num_segments=num_segments,
        res_mat=res_mat,
    )
    numpyro.factor("log_likelihood", ll)


# running the Bayesian model:
def mcmc_non_param(log_pdgrm, log_pdgrm_E, log_pdgrm_T,
         f, res_mat, n_knots=10, degree=3, penalty_order=1,
         num_segments=1,
         factor=1,
         phi_val=1e8, phi_val_oms=10000,
         phi_val_sgwb=10,
         num_warmup=500, num_samples=2500, num_chains=1, progress_bar=True,
         tm_spec=None,
         oms_spec=None,
         init_alpha=-3,
         init_log_10_omega=-13,
         ):
    '''
    MCMC function for SGWB separation from LISA noise. Signal and Noise models: Penalized P-splines
    :param log_pdgrm: log periodogram A channel
    :param log_pdgrm_E: log periodogram E channel
    :param log_pdgrm_T: log periodogram T channel
    :param f: frequencies
    :param res_mat: LISA GW response matrix
    :param n_knots: number of spline knots (assuming equal number of knots for all splines)
    :param degree: degree of the spline (default third order spline)
    :param penalty_order: penalty order of the penalty matrix (default 1)
    :param num_segments: number of segments
    :param factor: factor if normalized data
    :param phi_val: testmass spline parmeter (phi)
    :param phi_val_oms: oms spline parmeter (phi)
    :param phi_val_sgwb: SGWB spline parmeter (phi)
    :param num_warmup: number of warmup (NUTS)
    :param num_samples: number of samples (NUTS)
    :param num_chains: number of chains (NUTS)
    :param progress_bar: if show progress
    :param tm_spec: testmass theoretical spectrum (for initial spline weights)
    :param oms_spec: oms theoretical spectrum (for initial spline weights)
    :return: mcmc output object containing estimated results and plots.
    '''

    myobj_spl=init_var(n_knots=n_knots,
              f=f,
              degree=degree,
              penalty_order=penalty_order,
              tm_spec=tm_spec,
              oms_spec=oms_spec,
              factor=factor,
              log_pdgrm=log_pdgrm,
              log_pdgrm_E=log_pdgrm_E,
              log_pdgrm_T=log_pdgrm_T,
              res_mat=res_mat,
              np_sig=True,
              phi_val=phi_val,
              phi_val_oms=phi_val_oms,
              phi_val_sgwb=phi_val_sgwb,
              num_chains=num_chains,
              init_alpha=init_alpha,
              init_log_10_omega=init_log_10_omega,
    )
    nuts_kernel = NUTS(
        bayesian_model_non_param,
        init_strategy=init_to_value(values={
            "weights": myobj_spl.init_weights,
            "weights_oms": myobj_spl.init_weights_oms,
            "weights_sgwb" : myobj_spl.lambda_init_sgwb,
        }),
    )

    mcmc_bayes = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                      progress_bar=progress_bar)

    print("Running MCMC...")

    mcmc_bayes.run(
        jax.random.PRNGKey(123),
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        res_mat=myobj_spl.res_mat,
        num_segments=num_segments,
        factor=myobj_spl.factor,
        phi_val=phi_val,
        phi_val_oms=phi_val_oms,
        phi_val_sgwb=phi_val_sgwb,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        P_reg_sgwb=myobj_spl.P_reg_sgwb,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        weights_sgwb_loc=myobj_spl.weights_sgwb_loc,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
        basis_matrix_sgwb=myobj_spl.basis_matrix_sgwb,
    )
    out_obj=output(init_object=myobj_spl, sampler_object=mcmc_bayes, signal_param=False, signal_nparam=True)
    return out_obj







'''
Testing stepping stone sampling
'''
from scipy.stats import beta
def get_beta_ladder(n_betas=15):
    """
    Generate beta values using quantiles of Beta(0.3, 1).
    Concentrates points near 0 where the integrand varies most.
    """
    quantiles = np.linspace(0, 1, n_betas + 2)[1:-1]
    betas = beta.ppf(quantiles, a=0.3, b=1.0)
    betas = np.append(betas, 1.0)
    return np.sort(betas)

def bayesian_model_non_param_powered(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f, res_mat,
    num_segments,
    factor,
    phi_val,
    phi_val_oms,
    phi_val_sgwb,
    P_reg,
    P_reg_oms,
    P_reg_sgwb,
    init_weights,
    init_weights_oms,
    weights_sgwb_loc,
    basis_matrix,
    basis_matrix_oms,
    basis_matrix_sgwb,
    beta,
):
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )
    weights_sgwb = numpyro.sample(
        "weights_sgwb",
        dist.MultivariateNormal(loc=weights_sgwb_loc, precision_matrix=phi_val_sgwb * P_reg_sgwb),
    )

    ll = loglikelihood_nonparam(
        weights=weights,
        weights_oms=weights_oms,
        weights_sgwb=weights_sgwb,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        basis_matrix_sgwb=basis_matrix_sgwb,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        num_segments=num_segments,
        res_mat=res_mat,
    )
    # scale likelihood by beta
    numpyro.factor("log_likelihood", beta * ll)



def run_single_beta_nonparam(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f, res_mat,
    beta, beta_idx,
    n_knots=10, degree=3, penalty_order=1,
    num_segments=1, factor=1,
    phi_val=10000, phi_val_oms=10000, phi_val_sgwb=10,
    num_warmup=1000, num_samples=2000, num_chains=4,
    tm_spec=None, oms_spec=None,
):
    """
    Runs MCMC at a single beta value for the signal + noise model.
    """
    myobj_spl = init_var(
        n_knots=n_knots, f=f, degree=degree, penalty_order=penalty_order,
        tm_spec=tm_spec, oms_spec=oms_spec, factor=factor,
        log_pdgrm=log_pdgrm, log_pdgrm_E=log_pdgrm_E, log_pdgrm_T=log_pdgrm_T,
        res_mat=res_mat, np_sig=True,
        phi_val=phi_val, phi_val_oms=phi_val_oms, phi_val_sgwb=phi_val_sgwb,
        num_chains=num_chains,
    )

    nuts_kernel = NUTS(
        bayesian_model_non_param_powered,
        target_accept_prob=0.9,
        init_strategy=init_to_value(values={
            "weights":      myobj_spl.init_weights,
            "weights_oms":  myobj_spl.init_weights_oms,
            "weights_sgwb": myobj_spl.lambda_init_sgwb,
        }),
    )

    mcmc_b = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )

    mcmc_b.run(
        jax.random.PRNGKey(beta_idx),  # unique key per beta
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        res_mat=myobj_spl.res_mat,
        num_segments=num_segments,
        factor=factor,
        phi_val=phi_val,
        phi_val_oms=phi_val_oms,
        phi_val_sgwb=phi_val_sgwb,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        P_reg_sgwb=myobj_spl.P_reg_sgwb,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        weights_sgwb_loc=myobj_spl.weights_sgwb_loc,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
        basis_matrix_sgwb=myobj_spl.basis_matrix_sgwb,
        beta=float(beta),
    )

    samples = mcmc_b.get_samples()
    ll_samples = jax.vmap(
        lambda w, woms, wsgwb: loglikelihood_nonparam(
            weights=w,
            weights_oms=woms,
            weights_sgwb=wsgwb,
            basis_matrix=myobj_spl.basis_matrix,
            basis_matrix_oms=myobj_spl.basis_matrix_oms,
            basis_matrix_sgwb=myobj_spl.basis_matrix_sgwb,
            log_pdgrm=myobj_spl.log_pdgrm,
            log_pdgrm_E=myobj_spl.log_pdgrm_E,
            log_pdgrm_T=myobj_spl.log_pdgrm_T,
            f=myobj_spl.f,
            num_segments=num_segments,
            res_mat=myobj_spl.res_mat,
        )
    )(samples["weights"], samples["weights_oms"], samples["weights_sgwb"])

    ll_samples = np.asarray(ll_samples)
    print(f"  beta={beta:.5f} — mean ll={np.mean(ll_samples):.2f}, std={np.std(ll_samples):.2f}")
    return ll_samples



def bayesian_model_noise_powered(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,
    num_segments,
    phi_val,
    phi_val_oms,
    P_reg,
    P_reg_oms,
    init_weights,
    init_weights_oms,
    basis_matrix,
    basis_matrix_oms,
    beta,
):
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )

    ll = loglikelihood_noise(
        weights=weights,
        weights_oms=weights_oms,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        num_segments=num_segments,
    )
    numpyro.factor("log_likelihood", beta * ll)


def run_single_beta_noise(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,
    beta, beta_idx,
    n_knots=10, degree=3, penalty_order=1,
    num_segments=1, factor=1,
    phi_val=10000, phi_val_oms=10000,
    num_warmup=1000, num_samples=2000, num_chains=4,
    tm_spec=None, oms_spec=None,
):
    """
    Runs MCMC at a single beta value for the noise-only model.
    """
    myobj_spl = init_var(
        n_knots=n_knots, f=f, degree=degree, penalty_order=penalty_order,
        tm_spec=tm_spec, oms_spec=oms_spec, factor=factor,
        log_pdgrm=log_pdgrm, log_pdgrm_E=log_pdgrm_E, log_pdgrm_T=log_pdgrm_T,
        phi_val=phi_val, phi_val_oms=phi_val_oms,
        num_chains=num_chains,
    )

    nuts_kernel = NUTS(
        bayesian_model_noise_powered,
        target_accept_prob=0.9,
        init_strategy=init_to_value(values={
            "weights":     myobj_spl.init_weights,
            "weights_oms": myobj_spl.init_weights_oms,
        }),
    )

    mcmc_b = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )

    mcmc_b.run(
        jax.random.PRNGKey(beta_idx),
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        num_segments=num_segments,
        phi_val=phi_val,
        phi_val_oms=phi_val_oms,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
        beta=float(beta),
    )

    samples = mcmc_b.get_samples()
    ll_samples = jax.vmap(
        lambda w, woms: loglikelihood_noise(
            weights=w,
            weights_oms=woms,
            basis_matrix=myobj_spl.basis_matrix,
            basis_matrix_oms=myobj_spl.basis_matrix_oms,
            log_pdgrm=myobj_spl.log_pdgrm,
            log_pdgrm_E=myobj_spl.log_pdgrm_E,
            log_pdgrm_T=myobj_spl.log_pdgrm_T,
            f=myobj_spl.f,
            num_segments=num_segments,
        )
    )(samples["weights"], samples["weights_oms"])

    ll_samples = np.asarray(ll_samples)
    print(f"  beta={beta:.5f} — mean ll={np.mean(ll_samples):.2f}, std={np.std(ll_samples):.2f}")
    return ll_samples



def bayesian_model_param_powered(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f,res_mat,f0,
    num_segments,
    factor,
    phi_val,
    phi_val_oms,
    P_reg,
    P_reg_oms,
    init_weights,
    init_weights_oms,
    basis_matrix,
    basis_matrix_oms,
    beta,
):
    '''Bayesian model: parametric power law SGWB model and spline LISA noise model.'''
    weights = numpyro.sample(
        "weights",
        dist.MultivariateNormal(loc=init_weights, precision_matrix=phi_val * P_reg),
    )
    weights_oms = numpyro.sample(
        "weights_oms",
        dist.MultivariateNormal(loc=init_weights_oms, precision_matrix=phi_val_oms * P_reg_oms),
    )
    # Uniform priors
    log10_omega = numpyro.sample("log10_omega", dist.Uniform(-20.0, -9.0))
    omega_ref   = 10.0 ** log10_omega

    alpha_sgwb  = numpyro.sample("alpha_sgwb", dist.Uniform(-5.0, 5.0))

    ll = loglikelihood(
        weights=weights,
        weights_oms=weights_oms,
        basis_matrix=basis_matrix,
        basis_matrix_oms=basis_matrix_oms,
        log_pdgrm=log_pdgrm,
        log_pdgrm_E=log_pdgrm_E,
        log_pdgrm_T=log_pdgrm_T,
        f=f,
        omega_ref=omega_ref,
        alpha_sgwb=alpha_sgwb,
        num_segments=num_segments,
        res_mat=res_mat,
        f0=f0,
        factor=factor,
    )
    numpyro.factor("log_likelihood", beta * ll)







def run_single_beta_param(
    log_pdgrm, log_pdgrm_E, log_pdgrm_T,
    f, res_mat, f0,
    beta, beta_idx,
    n_knots=10, degree=3, penalty_order=1,
    num_segments=1, factor=1,
    phi_val=10000, phi_val_oms=10000,
    num_warmup=1000, num_samples=2000, num_chains=4,
    init_alpha=-3,
    init_log_10_omega=-13,
    tm_spec=None, oms_spec=None,
):
    """
    Runs MCMC at a single beta value for the signal + noise model.
    Returns unscaled log-likelihood values for each posterior draw.

    Parameters
    ----------
    beta      : float — the power posterior temperature for this run
    beta_idx  : int   — used as the PRNGKey seed so each job gets different randomness

    Returns
    -------
    ll_samples : np.ndarray, shape (num_samples * num_chains,)
        Unscaled log-likelihood values (at beta=1) for each posterior draw.
    """
    myobj_spl = init_var(n_knots=n_knots,
                         f=f,
                         degree=degree,
                         penalty_order=penalty_order,
                         tm_spec=tm_spec,
                         oms_spec=oms_spec,
                         factor=factor,
                         log_pdgrm=log_pdgrm,
                         log_pdgrm_E=log_pdgrm_E,
                         log_pdgrm_T=log_pdgrm_T,
                         f0=f0,
                         res_mat=res_mat,
                         phi_val=phi_val,
                         phi_val_oms=phi_val_oms,
                         num_chains=num_chains,
                         )

    nuts_kernel = NUTS(
        bayesian_model_param_powered,
        init_strategy=init_to_value(values={
            # IMPORTANT: correct variable names
            "alpha_sgwb": init_alpha,
            "log10_omega": init_log_10_omega,
            "weights": myobj_spl.init_weights,
            "weights_oms": myobj_spl.init_weights_oms,
        }),
    )

    mcmc_b = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
    )


    mcmc_b.run(
        jax.random.PRNGKey(beta_idx),  # unique key per beta
        log_pdgrm=myobj_spl.log_pdgrm,
        log_pdgrm_E=myobj_spl.log_pdgrm_E,
        log_pdgrm_T=myobj_spl.log_pdgrm_T,
        f=myobj_spl.f,
        res_mat=myobj_spl.res_mat,
        f0=myobj_spl.f0,
        num_segments=num_segments,
        factor=myobj_spl.factor,
        phi_val=myobj_spl.phi_val,
        phi_val_oms=myobj_spl.phi_val_oms,
        P_reg=myobj_spl.P_reg,
        P_reg_oms=myobj_spl.P_reg_oms,
        init_weights=myobj_spl.init_weights,
        init_weights_oms=myobj_spl.init_weights_oms,
        basis_matrix=myobj_spl.basis_matrix,
        basis_matrix_oms=myobj_spl.basis_matrix_oms,
        beta=float(beta),
    )

    samples = mcmc_b.get_samples()

    # Re-evaluate unscaled log-likelihood (beta=1) for each draw
    ll_samples = jax.vmap(
        lambda w, woms, nsgwb, log10omeg: loglikelihood(
            weights=w,
            weights_oms=woms,
            omega_ref=10.0 ** log10omeg,  # ← exponentiate here
            alpha_sgwb=nsgwb,
            basis_matrix=myobj_spl.basis_matrix,
            basis_matrix_oms=myobj_spl.basis_matrix_oms,
            log_pdgrm=myobj_spl.log_pdgrm,
            log_pdgrm_E=myobj_spl.log_pdgrm_E,
            log_pdgrm_T=myobj_spl.log_pdgrm_T,
            f=myobj_spl.f,
            num_segments=num_segments,
            res_mat=myobj_spl.res_mat,
            f0=f0,
            factor=factor,
        )
    )(samples["weights"], samples["weights_oms"], samples['alpha_sgwb'], samples["log10_omega"])

    ll_samples = np.asarray(ll_samples)
    print(f"  beta={beta:.5f} — mean ll={np.mean(ll_samples):.2f}, std={np.std(ll_samples):.2f}")
    return ll_samples







